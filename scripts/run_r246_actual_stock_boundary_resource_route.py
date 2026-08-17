"""Observe the remaining R235/R242/R244 stock-runtime facts, or fail closed.

This is an external, disposable preflight worker.  It never modifies the
submission tree, core runtime, contracts, a managed service, or a Kaggle
resource.  It proves three narrow facts from one *fresh*, exact staged
package:

* a real official-r236 ``SearchStep`` produces an actor-change successor and
  the staged frozen evaluator treats that successor as a value-only boundary;
* one literal staged parent/broker two-lane marker supplies the native R244
  handle/SearchId topology (with only the static namespace semantics projected
  from the exact r225 contract); and
* parent and exact broker-child RSS/thread/startup observations are measured
  while that broker child is still live.

The worker has no action authority.  Its one physical game is a source of
observations only, and its direct ``_evaluate_batch`` invocation does not
execute an action into that game.  Run it only inside the existing exact-child
watchdog, so a stock/native call that does not return is contained by the
owning preflight process.

``stdout`` is deliberately a single JSON object for
``run_r235_r246_exact_stage_probe.py``.  Package output is redirected to
``stderr`` while preserving the literal marker in that JSON object.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r228_kaggle_r244_harness_common import (
    COMPLETE_ACTION_CAP,
    as_action,
    collect_markers,
    load_binding_identity,
    prepare_exact_stage_import,
    require_module_from_exact_stage,
    sha256_file,
    stage_snapshot,
)

SCHEMA = "poke_bot.r235_r246_exact_stage_scenario_evidence/v1"
EVIDENCE_KIND = "actual_stock_actor_change_boundary_resource_startup_preflight_observation"
WITNESS_ORIGIN = "actual_stock_runtime_observation"
ACTUAL_RUNTIME_OBSERVATION_SCHEMA = (
    "poke_bot.r242_actual_stock_actor_change_runtime_observation/v1"
)
RESOURCE_OBSERVATION_SCHEMA = (
    "poke_bot.r238_actual_parent_broker_resource_startup_observation/v1"
)
R244_WITNESS_SCHEMA = "poke_bot.r244_handle_scoped_search_id_identity_probe/v1"
R244_OWNER_REVISION = 244
R238_STAGE_SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
STAGED_RUNTIME_MODULE_NAME = "poke_bot.r228_kaggle_async_runtime"
STAGED_RUNTIME_EVALUATOR_METHOD = "R228AsyncGameplay._evaluate_batch"
R244_WITNESS_ORIGIN = (
    "actual_staged_mcts_marker_topology_with_r225_contract_namespace_projection"
)
MAX_PARENT_STDOUT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_PHYSICAL_ACTIONS = 64


class ActualStockBoundaryResourceError(RuntimeError):
    """The exact staged package did not produce every required observation."""


@dataclass(frozen=True)
class _ProcessStatus:
    """One direct /proc status observation with explicit source units."""

    pid: int
    vm_rss_bytes: int
    vm_hwm_bytes: int
    thread_count: int
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "vm_rss_bytes": self.vm_rss_bytes,
            "vm_hwm_bytes": self.vm_hwm_bytes,
            "thread_count": self.thread_count,
            "source": self.source,
        }


@dataclass(frozen=True)
class _PhysicalBoundaryRoot:
    """One fresh physical root/action whose real game switched actors."""

    observation: dict[str, Any]
    action: list[int]
    root_actor_seat: int
    physical_successor: dict[str, Any]
    callback_elapsed_seconds: float
    callback_index: int


class _BoundedTee:
    """Forward staged output to stderr while retaining bounded marker text."""

    def __init__(self, target: TextIO, *, max_bytes: int) -> None:
        self._target = target
        self._max_bytes = int(max_bytes)
        self._parts: list[str] = []
        self._captured_bytes = 0
        self.truncated = False

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            value = str(value)
        self._target.write(value)
        encoded = value.encode("utf-8", errors="replace")
        remaining = self._max_bytes - self._captured_bytes
        if remaining > 0:
            captured = encoded[:remaining].decode("utf-8", errors="ignore")
            self._parts.append(captured)
            self._captured_bytes += len(captured.encode("utf-8"))
        if len(encoded) > max(0, remaining):
            self.truncated = True
        return len(value)

    def flush(self) -> None:
        self._target.flush()

    @property
    def text(self) -> str:
        return "".join(self._parts)


class _MarkerChannel:
    """A non-authoritative progress sink for one external evaluator instance."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def send(self, payload: Mapping[str, Any], *, deadline: float) -> None:
        # Retain exact progress payloads only for audit/debugging.  The worker
        # never treats a progress message as action authority.
        if not math.isfinite(float(deadline)):
            raise ActualStockBoundaryResourceError("runtime progress deadline is not finite")
        self.messages.append(_json_copy(dict(payload), label="runtime progress payload"))


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ActualStockBoundaryResourceError(f"{label} is not JSON-native") from exc


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActualStockBoundaryResourceError("marker is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActualStockBoundaryResourceError(f"{label} must be an object")
    return value


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActualStockBoundaryResourceError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ActualStockBoundaryResourceError(f"{label} must be at least {minimum}")
    return result


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActualStockBoundaryResourceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ActualStockBoundaryResourceError(f"{label} must be nonnegative and finite")
    return result


def _physical_directory(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ActualStockBoundaryResourceError(f"{label} must be an existing physical directory")
    return raw.resolve()


def _regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ActualStockBoundaryResourceError(f"{label} must be a regular non-symlink file")
    return raw.resolve()


def _stage_disk_bytes(stage: Path) -> int:
    total = 0
    for member in stage.rglob("*"):
        if member.is_symlink():
            raise ActualStockBoundaryResourceError("sealed stage contains a symlink")
        if member.is_file():
            total += member.stat().st_size
    return total


def _read_proc_status(pid: int) -> _ProcessStatus:
    """Read direct Linux process facts; absence is a hard failure, not a guess."""

    clean_pid = _integer(pid, label="process pid", minimum=1)
    status = Path("/proc") / str(clean_pid) / "status"
    try:
        lines = status.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ActualStockBoundaryResourceError(
            f"cannot read live process status for pid {clean_pid}"
        ) from exc
    values: dict[str, int] = {}
    for line in lines:
        name, separator, raw = line.partition(":")
        if not separator or name not in {"VmRSS", "VmHWM", "Threads"}:
            continue
        tokens = raw.split()
        if not tokens:
            continue
        try:
            values[name] = int(tokens[0])
        except ValueError as exc:
            raise ActualStockBoundaryResourceError(
                f"/proc status {name} is not numeric for pid {clean_pid}"
            ) from exc
    missing = {"VmRSS", "VmHWM", "Threads"}.difference(values)
    if missing:
        raise ActualStockBoundaryResourceError(
            "live /proc status lacks " + ", ".join(sorted(missing))
        )
    if values["VmRSS"] < 0 or values["VmHWM"] < values["VmRSS"] or values["Threads"] < 1:
        raise ActualStockBoundaryResourceError("live /proc status values are inconsistent")
    return _ProcessStatus(
        pid=clean_pid,
        vm_rss_bytes=values["VmRSS"] * 1024,
        vm_hwm_bytes=values["VmHWM"] * 1024,
        thread_count=values["Threads"],
        source="linux_proc_status_VmRSS_VmHWM_Threads_kib",
    )


def _raw_observation(value: Any, *, label: str) -> dict[str, Any]:
    """Convert one official API observation without accepting an arbitrary repr."""

    if isinstance(value, Mapping):
        raw = dict(value)
    elif is_dataclass(value):
        raw_value = asdict(value)
        if not isinstance(raw_value, dict):  # defensive; dataclass normally returns dict
            raise ActualStockBoundaryResourceError(f"{label} dataclass is not an object")
        raw = raw_value
    else:
        raise ActualStockBoundaryResourceError(
            f"{label} is not a stock observation mapping/dataclass"
        )
    copied = _json_copy(raw, label=label)
    if not isinstance(copied, dict):  # defensive after JSON copy
        raise ActualStockBoundaryResourceError(f"{label} is not a JSON object")
    return copied


def _actor_seat(observation: Mapping[str, Any], *, label: str) -> int:
    current = _mapping(observation.get("current"), label=f"{label} current")
    result = _integer(current.get("result"), label=f"{label} result")
    if result != -1:
        raise ActualStockBoundaryResourceError(f"{label} is already terminal")
    return _integer(current.get("yourIndex"), label=f"{label} actor", minimum=0)


def _deck(stage: Path) -> list[int]:
    cards: list[int] = []
    path = _regular_file(stage / "deck.csv", label="staged deck.csv")
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ActualStockBoundaryResourceError("staged deck.csv is unreadable") from exc
    for raw in rows:
        line = raw.strip()
        if line and not line.startswith("#"):
            try:
                cards.append(int(line.split(",", 1)[0]))
            except ValueError as exc:
                raise ActualStockBoundaryResourceError("staged deck contains a noninteger") from exc
        if len(cards) == 60:
            break
    if len(cards) != 60:
        raise ActualStockBoundaryResourceError("staged deck is not exactly 60 cards")
    return cards


def _complete_legal_actions(features: Any, observation: Mapping[str, Any]) -> list[list[int]]:
    try:
        raw = features.enumerate_action_combos(
            dict(observation), max_combos=COMPLETE_ACTION_CAP
        )
    except Exception as exc:
        raise ActualStockBoundaryResourceError(
            "staged complete action enumeration failed under the 65536 cap"
        ) from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ActualStockBoundaryResourceError("staged complete legal order is malformed")
    legal = [as_action(item, field="physical complete legal action") for item in raw]
    if not legal:
        raise ActualStockBoundaryResourceError("physical active prompt has no legal action")
    return legal


def _load_exact_stage(stage: Path) -> tuple[Any, Any, Any]:
    """Import only the sealed parent, stock wrapper, and feature module."""

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    stage = prepare_exact_stage_import(stage)
    os.environ["CG_LIB_PATH"] = str(stage)
    os.chdir(stage)
    main = importlib.import_module("main")
    require_module_from_exact_stage(main, module_name="main", stage=stage)
    cg_env = importlib.import_module("poke_bot.cg_env")
    features = importlib.import_module("poke_bot.features")
    require_module_from_exact_stage(cg_env, module_name="poke_bot.cg_env", stage=stage)
    require_module_from_exact_stage(features, module_name="poke_bot.features", stage=stage)
    return main, cg_env, features


def _marker_counts(text: str) -> dict[str, int]:
    return {kind: len(rows) for kind, rows in collect_markers(text).items()}


def _new_decision_marker(tee: _BoundedTee, *, prior: int) -> Mapping[str, Any] | None:
    if tee.truncated:
        raise ActualStockBoundaryResourceError("staged parent stdout marker capture was truncated")
    rows = collect_markers(tee.text).get("decisions", [])
    if len(rows) < prior:
        raise ActualStockBoundaryResourceError("staged decision marker count moved backwards")
    fresh = rows[prior:]
    if len(fresh) > 1:
        raise ActualStockBoundaryResourceError("one physical callback emitted multiple decision markers")
    return None if not fresh else _mapping(fresh[0], label="literal staged decision marker")


def _require_literal_two_lane_marker(
    marker: Mapping[str, Any], *, legal: Sequence[Sequence[int]], action: Sequence[int]
) -> dict[str, Any]:
    """Retain one actual, successful parent/broker MCTS marker verbatim."""

    result = dict(marker)
    if result.get("mode") != "shared_tree_mcts":
        raise ActualStockBoundaryResourceError("literal marker is not shared_tree_mcts")
    if result.get("degraded") is not False:
        raise ActualStockBoundaryResourceError("literal marker is degraded")
    if result.get("mcts_action_authority") is not True:
        raise ActualStockBoundaryResourceError("literal marker lacks MCTS action authority")
    if as_action(result.get("selected_action"), field="marker selected action") != list(action):
        raise ActualStockBoundaryResourceError("literal marker action differs from physical callback")
    normalized_legal = [list(row) for row in legal]
    if list(action) not in normalized_legal:
        raise ActualStockBoundaryResourceError("physical action is outside complete legal order")
    for field, expected in (
        ("requested_simulator_lane_count", 2),
        ("active_simulator_lane_count", 2),
        ("arena_count", 2),
        ("unique_handle_count", 2),
        ("search_begin_calls", 2),
        ("configured_simulator_lane_count", 2),
    ):
        if _integer(result.get(field), label=f"literal marker {field}", minimum=0) != expected:
            raise ActualStockBoundaryResourceError(f"literal marker {field} is not two")
    broker = _mapping(result.get("broker"), label="literal marker broker")
    if broker.get("child_pid") is None:
        raise ActualStockBoundaryResourceError("literal marker has no live broker child pid")
    _integer(broker.get("child_pid"), label="literal broker child pid", minimum=1)
    if result.get("parent_cuda_runtime_before_search") is None:
        raise ActualStockBoundaryResourceError("literal marker lacks parent CUDA observation")
    return result


def _close_exact_broker(main: Any) -> None:
    broker = getattr(main, "_BROKER", None)
    if broker is None:
        raise ActualStockBoundaryResourceError("sealed parent lost its broker before exact close")
    close = getattr(broker, "close", None)
    if not callable(close):
        raise ActualStockBoundaryResourceError("sealed parent broker lacks close")
    close()
    state = getattr(broker, "marker_payload", None)
    if callable(state):
        observed = _mapping(state(), label="broker post-close state")
        if observed.get("child_pid") is not None:
            raise ActualStockBoundaryResourceError("exact broker child remains live after close")


def _capture_mcts_marker_and_physical_actor_change(
    *,
    main: Any,
    cg_env: Any,
    features: Any,
    deck: Sequence[int],
    max_physical_actions: int,
    tee: _BoundedTee,
) -> tuple[dict[str, Any], _PhysicalBoundaryRoot]:
    """Drive one fresh stock game until MCTS marker and a real actor change occur."""

    if max_physical_actions <= 0:
        raise ActualStockBoundaryResourceError("max physical actions must be positive")
    observation, start_result = cg_env.battle_start(list(deck), list(deck))
    if observation is None:
        raise ActualStockBoundaryResourceError(
            f"official stock BattleStart failed: {getattr(start_result, 'errorType', None)}"
        )
    raw_observation = _raw_observation(observation, label="fresh physical BattleStart observation")
    observed_mcts_marker = False
    try:
        for callback_index in range(max_physical_actions):
            if cg_env.is_finished(raw_observation):
                break
            root_actor = _actor_seat(raw_observation, label="fresh physical root")
            legal = _complete_legal_actions(features, raw_observation)
            prior = _marker_counts(tee.text).get("decisions", 0)
            started = time.monotonic()
            with contextlib.redirect_stdout(tee):
                action = as_action(main.agent(dict(raw_observation)), field="sealed parent action")
            callback_elapsed = _finite_nonnegative(
                time.monotonic() - started, label="physical callback elapsed"
            )
            if action not in legal:
                raise ActualStockBoundaryResourceError(
                    "sealed parent returned an action outside complete physical legal order"
                )
            marker = _new_decision_marker(tee, prior=prior)
            callback_marker: dict[str, Any] | None = None
            if marker is not None and marker.get("mode") == "shared_tree_mcts":
                candidate = _require_literal_two_lane_marker(marker, legal=legal, action=action)
                observed_mcts_marker = True
                # The actor-change root must be this same stock-accepted
                # callback: the outer route binds the literal two-lane marker
                # action to the raw official SearchStep action byte-for-byte.
                callback_marker = candidate
            successor = _raw_observation(
                cg_env.battle_select(list(action)), label="fresh physical BattleSelect successor"
            )
            if not cg_env.is_finished(successor):
                successor_actor = _actor_seat(successor, label="fresh physical successor")
                if successor_actor != root_actor:
                    if callback_marker is None:
                        # A direct or prior-MCTS callback cannot be silently
                        # relabelled as this native SearchStep source.  Keep
                        # driving until one *same-callback* MCTS action
                        # reaches a real actor change.
                        raw_observation = successor
                        continue
                    return callback_marker, _PhysicalBoundaryRoot(
                        observation=dict(raw_observation),
                        action=list(action),
                        root_actor_seat=root_actor,
                        physical_successor=successor,
                        callback_elapsed_seconds=callback_elapsed,
                        callback_index=callback_index,
                    )
            raw_observation = successor
    finally:
        finish = getattr(cg_env, "battle_finish", None)
        if callable(finish):
            try:
                finish()
            except Exception as exc:
                raise ActualStockBoundaryResourceError("fresh physical BattleFinish failed") from exc
    if not observed_mcts_marker:
        raise ActualStockBoundaryResourceError("no literal successful two-lane MCTS marker was observed")
    raise ActualStockBoundaryResourceError(
        "fresh physical route did not reach an actor-change/end-turn boundary on the same MCTS callback"
    )


def _search_step_actor_change_observation(
    *,
    stage: Path,
    binding: Mapping[str, Any],
    root: _PhysicalBoundaryRoot,
    cg_env: Any,
    deck: Sequence[int],
) -> dict[str, Any]:
    """Run exactly one official-r236 raw SearchStep from the observed root."""

    lane_module = importlib.import_module("poke_bot.r225_stock_native_lane")
    runtime_module = importlib.import_module(STAGED_RUNTIME_MODULE_NAME)
    require_module_from_exact_stage(
        lane_module, module_name="poke_bot.r225_stock_native_lane", stage=stage
    )
    require_module_from_exact_stage(
        runtime_module, module_name=STAGED_RUNTIME_MODULE_NAME, stage=stage
    )
    lane_type = getattr(lane_module, "R225StockNativeSearchLane", None)
    prewarm = getattr(lane_module, "prewarm_stock_cg", None)
    canonical_fingerprint = getattr(runtime_module, "canonical_observation_fingerprint", None)
    if not callable(lane_type) or not callable(prewarm) or not callable(canonical_fingerprint):
        raise ActualStockBoundaryResourceError(
            "sealed stage lacks the official native-lane/fingerprint interfaces"
        )
    common = _mapping(binding.get("common_identity"), label="binding common identity")
    expected_libcg_sha256 = common.get("linux_x86_64_libcg_sha256")
    if not isinstance(expected_libcg_sha256, str) or not expected_libcg_sha256:
        raise ActualStockBoundaryResourceError("exact binding lacks Linux r236 libcg SHA-256")
    actual_libcg_sha256 = sha256_file(
        _regular_file(stage / "cg" / "libcg.so", label="staged official Linux libcg")
    )
    if actual_libcg_sha256 != expected_libcg_sha256:
        raise ActualStockBoundaryResourceError(
            "staged official Linux libcg SHA-256 differs from exact binding"
        )
    if Path(os.environ.get("CG_LIB_PATH", "")).resolve() != stage:
        raise ActualStockBoundaryResourceError("official SearchStep is not bound to the sealed stage")
    if not root.action:
        raise ActualStockBoundaryResourceError(
            "physical actor-change root has no selected action for official SearchStep"
        )

    api, sim = prewarm()
    lane = lane_type(0, lib=sim.lib, api_module=api)
    search_inputs = cg_env.build_search_inputs(
        dict(root.observation), list(deck), opponent_deck_guess=list(deck)
    )
    root_state: Any | None = None
    successor_state: Any | None = None
    try:
        root_state = lane.search_begin(root.observation, search_inputs, manual_coin=True)
        root_search_id = _integer(
            getattr(root_state, "searchId", None), label="official SearchBegin SearchId", minimum=0
        )
        successor_state = lane.search_step(root_search_id, root.action)
        successor_search_id = _integer(
            getattr(successor_state, "searchId", None), label="official SearchStep SearchId", minimum=0
        )
        successor = _raw_observation(
            getattr(successor_state, "observation", None),
            label="official r236 SearchStep actor-change successor",
        )
    finally:
        # Release every exact state opened through this handle, then end only
        # this worker-owned handle.  Never touch another process/session.
        for search_id in sorted(getattr(lane, "live_search_ids", ())):
            lane.search_release(int(search_id))
        lane.search_end()
    leaf_actor = _actor_seat(successor, label="official SearchStep successor")
    if leaf_actor == root.root_actor_seat:
        raise ActualStockBoundaryResourceError(
            "official SearchStep successor did not change away from root actor"
        )
    physical_actor = _actor_seat(root.physical_successor, label="physical actor-change successor")
    if physical_actor != leaf_actor:
        raise ActualStockBoundaryResourceError(
            "official SearchStep and physical successor actor identities differ"
        )
    handle_identity = lane.handle_identity
    if isinstance(handle_identity, bool) or not isinstance(handle_identity, (int, str)):
        raise ActualStockBoundaryResourceError("official SearchStep handle identity is malformed")
    if isinstance(handle_identity, str) and not handle_identity:
        raise ActualStockBoundaryResourceError("official SearchStep handle identity is empty")
    search_step = {
        "search_begin_succeeded": True,
        "search_step_succeeded": True,
        "lane_handle_identity": handle_identity,
        "root_search_id": root_search_id,
        "selected_action": list(root.action),
        "root_actor_seat": root.root_actor_seat,
        "successor_actor_seat": leaf_actor,
        "official_linux_x86_64_libcg_sha256": actual_libcg_sha256,
    }
    return {
        "observation_origin": "fresh_official_r236_search_step_actor_change_successor",
        "official_r236_search_step_succeeded": True,
        "official_r236_search_step": search_step,
        "official_search_step_call_count": 1,
        "official_search_handle_identity": handle_identity,
        "official_search_begin_root_search_id": root_search_id,
        "official_search_step_successor_search_id": successor_search_id,
        "manual_coin": True,
        "root_observation_fingerprint": canonical_fingerprint(root.observation),
        "successor_observation_fingerprint": canonical_fingerprint(successor),
        "successor_observation": successor,
        "root_actor_seat": root.root_actor_seat,
        "leaf_actor_seat": leaf_actor,
    }


def _evaluate_actual_boundary_leaf(
    *,
    stage: Path,
    leaf_source: Mapping[str, Any],
    deck: Sequence[int],
) -> dict[str, Any]:
    """Invoke the sealed frozen evaluator once on the real SearchStep leaf."""

    broker_module = importlib.import_module("poke_bot.r228_kaggle_broker")
    runtime_module = importlib.import_module(STAGED_RUNTIME_MODULE_NAME)
    require_module_from_exact_stage(broker_module, module_name="poke_bot.r228_kaggle_broker", stage=stage)
    require_module_from_exact_stage(
        runtime_module, module_name=STAGED_RUNTIME_MODULE_NAME, stage=stage
    )
    new_runtime = getattr(broker_module, "_child_new_runtime", None)
    frontier_type = getattr(runtime_module, "_Frontier", None)
    if not callable(new_runtime) or frontier_type is None:
        raise ActualStockBoundaryResourceError("sealed runtime lacks exact child evaluator interfaces")
    channel = _MarkerChannel()
    runtime = new_runtime(stage, channel)
    close = getattr(runtime, "close", None)
    if not callable(close):
        raise ActualStockBoundaryResourceError("sealed child runtime lacks close")
    root_actor = _integer(leaf_source.get("root_actor_seat"), label="leaf root actor", minimum=0)
    leaf_actor = _integer(leaf_source.get("leaf_actor_seat"), label="leaf actor", minimum=0)
    successor = _mapping(leaf_source.get("successor_observation"), label="actual SearchStep successor")
    try:
        policy = getattr(runtime, "policy", None)
        if policy is None:
            raise ActualStockBoundaryResourceError("sealed child runtime lacks frozen policy")
        route_getter = getattr(policy, "_matchup_model_route", None)
        if not callable(route_getter):
            raise ActualStockBoundaryResourceError("sealed frozen policy lacks matchup route")
        route = _integer(route_getter(), label="sealed frozen matchup route", minimum=0)
        opponent = tuple(int(card) for card in deck)
        runtime._decision = {
            "root_seat": root_actor,
            "route": route,
            "opponent_deck": opponent,
            "history_boards": [],
            "history_previous_actions": [],
            "boundary_leaf_count": 0,
            "chance_boundary_leaf_count": 0,
            "actor_change_boundary_leaf_count": 0,
        }
        frontier = frontier_type(lane_id=0, raw=dict(successor))
        leaves = runtime._evaluate_batch([frontier])
        if not isinstance(leaves, Sequence) or len(leaves) != 1:
            raise ActualStockBoundaryResourceError("sealed frozen evaluator did not return one leaf")
        leaf = leaves[0]
        value = float(getattr(leaf, "value", float("nan")))
        if not math.isfinite(value):
            raise ActualStockBoundaryResourceError("frozen evaluator value is nonfinite")
        if getattr(leaf, "boundary", None) is not True:
            raise ActualStockBoundaryResourceError("sealed evaluator did not close actor-change leaf")
        if getattr(leaf, "actor_change_boundary", None) is not True:
            raise ActualStockBoundaryResourceError("sealed evaluator did not classify actor-change boundary")
        if getattr(leaf, "chance_boundary", None) is not False:
            raise ActualStockBoundaryResourceError("actor-change witness crossed a chance boundary")
        if getattr(leaf, "unresolved_randomness", None) is not False:
            raise ActualStockBoundaryResourceError("actor-change witness has unresolved randomness")
        if _integer(getattr(leaf, "actor_seat", None), label="sealed leaf actor", minimum=0) != leaf_actor:
            raise ActualStockBoundaryResourceError("sealed evaluator leaf actor differs from SearchStep")
        legal_actions = getattr(leaf, "legal_actions", None)
        priors = getattr(leaf, "priors", None)
        if legal_actions != () or priors != ():
            raise ActualStockBoundaryResourceError("sealed evaluator materialized boundary actions/priors")
        context = _mapping(getattr(runtime, "_decision", None), label="sealed evaluator context")
        if _integer(context.get("actor_change_boundary_leaf_count"), label="actor boundary count", minimum=0) != 1:
            raise ActualStockBoundaryResourceError("sealed evaluator did not count exactly one actor boundary")
        if _integer(context.get("boundary_leaf_count"), label="boundary leaf count", minimum=0) != 1:
            raise ActualStockBoundaryResourceError("sealed evaluator did not close exactly one boundary")
        if _integer(context.get("chance_boundary_leaf_count"), label="chance boundary count", minimum=0) != 0:
            raise ActualStockBoundaryResourceError("actor boundary evaluator also counted chance")
        module_file = getattr(runtime_module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ActualStockBoundaryResourceError("sealed runtime module lacks a physical path")
        return {
            "schema": ACTUAL_RUNTIME_OBSERVATION_SCHEMA,
            "observation_origin": leaf_source["observation_origin"],
            "sealed_stage_runtime_module": STAGED_RUNTIME_MODULE_NAME,
            "sealed_stage_runtime_module_path": str(Path(module_file).resolve()),
            "sealed_runtime_evaluator_method": STAGED_RUNTIME_EVALUATOR_METHOD,
            "root_actor_seat": root_actor,
            "leaf_actor_seat": leaf_actor,
            "root_observation_fingerprint": leaf_source["root_observation_fingerprint"],
            "successor_observation_fingerprint": leaf_source["successor_observation_fingerprint"],
            "official_r236_search_step_succeeded": leaf_source[
                "official_r236_search_step_succeeded"
            ],
            "official_r236_search_step": _json_copy(
                leaf_source["official_r236_search_step"],
                label="actual official r236 SearchStep observation",
            ),
            "official_search_step_call_count": leaf_source["official_search_step_call_count"],
            "official_search_handle_identity": leaf_source["official_search_handle_identity"],
            "official_search_begin_root_search_id": leaf_source["official_search_begin_root_search_id"],
            "official_search_step_successor_search_id": leaf_source[
                "official_search_step_successor_search_id"
            ],
            "frozen_evaluator_value_call_count": 1,
            "model_value_evaluated": True,
            "frozen_evaluator_value": float(leaf.value),
            "expanded_legal_action_count": 0,
            "expanded_child_count": 0,
            "search_steps_beyond_boundary": 0,
            "opponent_action_selected_or_planned": False,
            "opponent_action_cached": False,
            "leaf_boundary": True,
            "leaf_actor_change_boundary": True,
            "leaf_chance_boundary": False,
            "leaf_unresolved_randomness": False,
            "action_authority_granted": False,
            "runtime_progress_message_count": len(channel.messages),
        }
    finally:
        runtime._decision = None
        close()


def _r244_witness_from_literal_marker(
    *, marker: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve native topology verbatim; identify static contract projection."""

    common = _mapping(binding.get("common_identity"), label="binding common identity")
    exact = _mapping(binding.get("exact_package"), label="binding exact package")
    stage_contract = _mapping(binding.get("stage_contract"), label="binding stage contract")
    topology_fields = (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "per_lane_handle_identities",
        "per_lane_search_id_chains",
        "per_lane_first_search_ids",
        "handle_scoped_first_search_id_composite_states",
    )
    witness: dict[str, Any] = {
        "schema": R244_WITNESS_SCHEMA,
        "witness_origin": R244_WITNESS_ORIGIN,
    }
    for field in topology_fields:
        if field not in marker:
            raise ActualStockBoundaryResourceError(f"literal marker lacks R244 topology {field}")
        witness[field] = _json_copy(marker[field], label=f"literal marker {field}")
    # These are contract semantics, never presented as a native observation.
    witness.update(
        {
            "search_id_numeric_namespace": "per_distinct_agent_start_handle",
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "semantic_contract_source": {
                "kind": "r225_r244_static_handle_namespace_contract_projection",
                "r225_contract_sha256": common.get("r225_contract_sha256"),
                "owner_handle_scoped_search_id_revision": R244_OWNER_REVISION,
                "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
                "globally_distinct_raw_search_id_integers_required": False,
                "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            },
            "literal_staged_marker_sha256": _canonical_sha256(marker),
            "literal_staged_marker": _json_copy(dict(marker), label="literal staged marker"),
            "common_identity": _json_copy(dict(common), label="R244 common identity"),
            "exact_package_identity": _json_copy(dict(exact), label="R244 exact package identity"),
            "stage_contract": _json_copy(dict(stage_contract), label="R244 stage contract"),
        }
    )
    return witness


def _resource_probe_from_actual_measurements(
    *,
    stage_disk_bytes: int,
    marker: Mapping[str, Any],
    binding: Mapping[str, Any],
    parent: _ProcessStatus,
    broker_child: _ProcessStatus,
    startup_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build resource fields from direct parent/broker observations only."""

    common = _mapping(binding.get("common_identity"), label="binding common identity")
    target = _mapping(common.get("phase1_submission_environment"), label="phase1 target")
    if target.get("vcpus") != 2:
        raise ActualStockBoundaryResourceError("exact binding is not the two-vCPU Phase-1 target")
    broker = _mapping(marker.get("broker"), label="literal marker broker")
    child_identity = _mapping(
        broker.get("child_identity"), label="literal marker broker child identity"
    )
    if _integer(broker.get("child_pid"), label="literal broker child pid", minimum=1) != broker_child.pid:
        raise ActualStockBoundaryResourceError(
            "literal marker broker pid differs from the measured exact child"
        )
    if _integer(child_identity.get("pid"), label="literal broker child identity pid", minimum=1) != broker_child.pid:
        raise ActualStockBoundaryResourceError(
            "literal marker child identity differs from the measured exact child"
        )
    _finite_nonnegative(
        child_identity.get("started_monotonic"), label="literal broker child ready startup clock"
    )
    parent_cuda = _mapping(
        marker.get("parent_cuda_runtime_before_search"),
        label="literal marker parent CUDA observation",
    )
    broker_child_cuda = _mapping(
        child_identity.get("cuda_runtime_before_search"),
        label="literal marker broker-child CUDA observation",
    )
    # `/proc/<pid>/status` is the authoritative operating-system observation
    # for process thread high-water values.  Do not relabel native lane count
    # as OS thread count; an over-budget actual observation must fail later
    # rather than being hidden by a two-lane receipt.
    observed_worker_threads = max(parent.thread_count, broker_child.thread_count)
    marker_runtime = {
        "configured_vcpus": int(target["vcpus"]),
        "configured_simulator_lane_count": _integer(
            marker.get("configured_simulator_lane_count"), label="marker configured lanes", minimum=0
        ),
        "maximum_simulator_lanes": _integer(
            marker.get("requested_simulator_lane_count"), label="marker requested lanes", minimum=0
        ),
        "observed_active_simulator_lane_count": _integer(
            marker.get("active_simulator_lane_count"), label="marker active lanes", minimum=0
        ),
        "receipt_lane_count": _integer(
            marker.get("arena_count"), label="marker arena count", minimum=0
        ),
        "receipt_schema": R238_STAGE_SCHEMA,
        "worker_thread_count": observed_worker_threads,
        "observed_peak_worker_threads": observed_worker_threads,
        "maximum_simulator_calls_in_flight": _integer(
            marker.get("max_simulator_calls_in_flight"), label="marker max in-flight", minimum=0
        ),
    }
    if any(
        marker_runtime[field] != 2
        for field in (
            "configured_vcpus",
            "configured_simulator_lane_count",
            "maximum_simulator_lanes",
            "observed_active_simulator_lane_count",
            "receipt_lane_count",
        )
    ):
        raise ActualStockBoundaryResourceError("literal staged marker does not prove the exact two-lane Phase-1 runtime")
    if marker_runtime["maximum_simulator_calls_in_flight"] > 2:
        raise ActualStockBoundaryResourceError("literal staged marker oversubscribed simulator calls")
    combined_nested_peak = parent.vm_hwm_bytes + broker_child.vm_hwm_bytes
    raw = {
        "schema": RESOURCE_OBSERVATION_SCHEMA,
        "measurement_origin": "fresh_sealed_parent_and_exact_broker_child",
        "measurement_phase": "literal_staged_parent_broker_marker_before_exact_broker_close",
        # The broker's child identity is populated only after its actual
        # ready IPC succeeds; this observation is taken while its exact PID
        # still has a readable /proc status record.
        "startup_ready_before_first_search": True,
        "broker_child_observed_while_alive": True,
        "startup_seconds": startup_seconds,
        "runtime_disk_bytes": stage_disk_bytes,
        "phase1_target": _json_copy(dict(target), label="phase1 target"),
        "configured_vcpus": marker_runtime["configured_vcpus"],
        "configured_simulator_lane_count": marker_runtime["configured_simulator_lane_count"],
        "maximum_simulator_lanes": marker_runtime["maximum_simulator_lanes"],
        "observed_active_simulator_lane_count": marker_runtime[
            "observed_active_simulator_lane_count"
        ],
        "receipt_lane_count": marker_runtime["receipt_lane_count"],
        "receipt_schema": marker_runtime["receipt_schema"],
        "maximum_simulator_calls_in_flight": marker_runtime[
            "maximum_simulator_calls_in_flight"
        ],
        "parent_peak_rss_bytes": parent.vm_hwm_bytes,
        "broker_child_peak_rss_bytes": broker_child.vm_hwm_bytes,
        "parent_worker_thread_count_peak": parent.thread_count,
        "broker_child_worker_thread_count_peak": broker_child.thread_count,
        "combined_nested_parent_broker_peak_rss_bytes": combined_nested_peak,
        "combined_nested_parent_broker_peak_formula": (
            "parent_proc_status_VmHWM_bytes + broker_child_proc_status_VmHWM_bytes"
        ),
        "parent_process": parent.as_dict(),
        "broker_child_process": broker_child.as_dict(),
        "runtime_lane_measurement_source": "literal_staged_shared_tree_mcts_marker",
        "parent_cuda_runtime_before_search": _json_copy(
            dict(parent_cuda), label="literal parent CUDA observation"
        ),
        "broker_child_cuda_runtime_before_search": _json_copy(
            dict(broker_child_cuda), label="literal broker-child CUDA observation"
        ),
        "literal_staged_marker_sha256": _canonical_sha256(marker),
    }
    probe = {
        "runtime_disk_bytes": stage_disk_bytes,
        # The outer exact-child watchdog records this whole worker as one
        # child.  Include both nested parent and broker high-water marks so
        # the later sum is conservative rather than silently omitting broker
        # memory.
        "child_peak_rss_bytes": combined_nested_peak,
        "phase1_target": _json_copy(dict(target), label="phase1 target"),
        "runtime": marker_runtime,
        "cuda_runtime_before_search": _json_copy(
            dict(parent_cuda), label="marker CUDA observation"
        ),
        "resource_observation_source": raw,
    }
    return probe, raw


def _write_json_once(
    path: Path, payload: Mapping[str, Any], *, sealed_stage: Path | None = None
) -> None:
    raw_target = Path(path).expanduser()
    if not raw_target.is_absolute():
        raise ActualStockBoundaryResourceError(
            "write-once witness output must be an absolute path outside the sealed stage"
        )
    if raw_target.exists() or raw_target.is_symlink():
        raise ActualStockBoundaryResourceError(
            f"write-once witness output already exists: {raw_target}"
        )
    target = raw_target.resolve()
    if sealed_stage is not None:
        try:
            target.relative_to(sealed_stage.resolve())
        except ValueError:
            pass
        else:
            raise ActualStockBoundaryResourceError(
                "write-once witness output must be outside the sealed stage"
            )
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ActualStockBoundaryResourceError("write-once witness parent is not physical")
    temporary = parent / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise ActualStockBoundaryResourceError("write-once witness temporary path already exists")
    encoded = json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ActualStockBoundaryResourceError(
                f"write-once witness output already exists: {target}"
            ) from exc
        os.chmod(target, 0o444)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _run(
    *,
    stage: Path,
    candidate_archive: Path,
    member_manifest: Path,
    r225_contract: Path,
    r236_contract: Path,
    max_physical_actions: int,
    r244_witness_output: Path | None,
) -> dict[str, Any]:
    stage = _physical_directory(stage, label="sealed stage")
    archive = _regular_file(candidate_archive, label="candidate archive")
    manifest = _regular_file(member_manifest, label="candidate member manifest")
    r225 = _regular_file(r225_contract, label="canonical r225 contract")
    r236 = _regular_file(r236_contract, label="canonical r236 contract")
    before = stage_snapshot(stage)
    started = time.monotonic()
    binding = load_binding_identity(
        stage=stage,
        candidate_archive=archive,
        member_manifest=manifest,
        r225_contract=r225,
        r236_contract=r236,
    )
    stage_disk_bytes = _stage_disk_bytes(stage)
    main, cg_env, features = _load_exact_stage(stage)
    deck = _deck(stage)
    tee = _BoundedTee(sys.stderr, max_bytes=MAX_PARENT_STDOUT_BYTES)
    marker: dict[str, Any] | None = None
    boundary_root: _PhysicalBoundaryRoot | None = None
    parent_status: _ProcessStatus | None = None
    broker_status: _ProcessStatus | None = None
    try:
        marker, boundary_root = _capture_mcts_marker_and_physical_actor_change(
            main=main,
            cg_env=cg_env,
            features=features,
            deck=deck,
            max_physical_actions=max_physical_actions,
            tee=tee,
        )
        broker = _mapping(marker.get("broker"), label="literal marker broker")
        child_pid = _integer(broker.get("child_pid"), label="literal broker child pid", minimum=1)
        parent_status = _read_proc_status(os.getpid())
        broker_status = _read_proc_status(child_pid)
        startup_seconds = _finite_nonnegative(
            time.monotonic() - started,
            label="actual stage-import-to-first-MCTS-route startup seconds",
        )
        # The direct search/evaluator measurement happens only after the
        # parent broker is cleanly closed, so no external worker introduces
        # concurrent native calls into the live broker process.
        _close_exact_broker(main)
        search_leaf = _search_step_actor_change_observation(
            stage=stage,
            binding=binding,
            root=boundary_root,
            cg_env=cg_env,
            deck=deck,
        )
        runtime_observation = _evaluate_actual_boundary_leaf(
            stage=stage, leaf_source=search_leaf, deck=deck
        )
    finally:
        # If a pre-marker failure occurs, close only the broker object this
        # disposable worker started.  Never signal a group or an unrelated
        # session.  A close failure remains a fail-closed worker failure.
        broker = getattr(main, "_BROKER", None)
        if broker is not None:
            close = getattr(broker, "close", None)
            if callable(close):
                close()
    after = stage_snapshot(stage)
    if before["tree_sha256"] != after["tree_sha256"]:
        raise ActualStockBoundaryResourceError("sealed stage changed during stock observation")
    if marker is None or boundary_root is None or parent_status is None or broker_status is None:
        raise ActualStockBoundaryResourceError("actual stock worker did not complete required observations")
    actor_runtime = _mapping(runtime_observation, label="actual stock runtime observation")
    actor_runtime = dict(actor_runtime)
    actor_runtime["stage_mutation_unchanged"] = True
    actor_runtime["stage_tree_sha256"] = before["tree_sha256"]
    resource_probe, resource_observation = _resource_probe_from_actual_measurements(
        stage_disk_bytes=stage_disk_bytes,
        marker=marker,
        binding=binding,
        parent=parent_status,
        broker_child=broker_status,
        startup_seconds=startup_seconds,
    )
    r244_witness = _r244_witness_from_literal_marker(marker=marker, binding=binding)
    if r244_witness_output is not None:
        _write_json_once(r244_witness_output, r244_witness, sealed_stage=stage)
    return {
        "schema": SCHEMA,
        "status": "passed",
        "passed": True,
        "witness_origin": WITNESS_ORIGIN,
        "evidence_kind": EVIDENCE_KIND,
        "common_identity": _json_copy(binding["common_identity"], label="common identity"),
        "exact_package_identity": _json_copy(binding["exact_package"], label="exact package identity"),
        "stage_contract": _json_copy(binding["stage_contract"], label="stage contract"),
        "literal_staged_marker": _json_copy(marker, label="literal staged marker"),
        "literal_staged_marker_sha256": _canonical_sha256(marker),
        "physical_stock_callback": {
            "callback_index": boundary_root.callback_index,
            "stock_action_accepted": True,
            "action": list(boundary_root.action),
            "callback_elapsed_seconds": boundary_root.callback_elapsed_seconds,
            "root_actor_seat": boundary_root.root_actor_seat,
        },
        "actual_stock_runtime_observation": actor_runtime,
        "actual_parent_broker_resource_startup_observation": resource_observation,
        "observed_resource_probe": resource_probe,
        "startup_seconds": startup_seconds,
        "r244_literal_witness": r244_witness,
        # R244's three namespace booleans are static r225 contract semantics;
        # all topology vectors above remain verbatim native marker evidence.
        "semantic_contract_source": _json_copy(
            r244_witness["semantic_contract_source"],
            label="R244 static contract projection",
        ),
        "stage_mutation_check": {
            "before_tree_sha256": before["tree_sha256"],
            "after_tree_sha256": after["tree_sha256"],
            "unchanged": True,
        },
        "action_authority_granted": False,
        "network_accessed": False,
        "kaggle_api_called": False,
        "kaggle_upload_used": False,
        "kaggle_queue_used": False,
        "bo_workload_started": False,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument("--r225-contract", type=Path, required=True)
    parser.add_argument("--r236-contract", type=Path, required=True)
    parser.add_argument(
        "--max-physical-actions", type=_positive_int, default=DEFAULT_MAX_PHYSICAL_ACTIONS
    )
    parser.add_argument(
        "--r244-witness-output",
        type=Path,
        help=(
            "optional absolute write-once standalone actual R244 witness JSON "
            "outside the sealed stage"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Staged stdout can contain decision markers.  The outer caller needs
        # exactly one typed JSON object on our stdout, so all helper output
        # stays on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            payload = _run(
                stage=args.stage,
                candidate_archive=args.candidate_archive,
                member_manifest=args.member_manifest,
                r225_contract=args.r225_contract,
                r236_contract=args.r236_contract,
                max_physical_actions=args.max_physical_actions,
                r244_witness_output=args.r244_witness_output,
            )
    except Exception as exc:  # noqa: BLE001 - one-object stdout protocol
        print(
            "R246_ACTUAL_STOCK_BOUNDARY_RESOURCE_ROUTE_FAILED_CLOSED "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
