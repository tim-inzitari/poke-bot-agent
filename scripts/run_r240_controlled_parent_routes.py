#!/usr/bin/env python3
"""Exercise selected R240/R242 parent routes with explicit controlled inputs.

This helper is intentionally package-external and *not* a physical-game,
simulator, model, CUDA, or preflight evidence producer.  It imports ``main``
and ``poke_bot.features`` only from a supplied sealed stage, preserves the
parent's real staged-stock-library identity check, and then installs temporary
in-memory adapters for the frozen direct policy, complete legal order, and
broker response.  The staged parent's own ``agent`` function therefore owns
the route selection, history rewrite, journal, marker emission, and receipt
validation under test.

The only supported routes are deliberately narrow:

* R242 inclusive 0.80 high-confidence direct with no broker construction;
* an actual parent deterministic-continuation consume after a controlled
  two-lane receipt; and
* independently controlled chance, fingerprint, illegal-action, and
  actor-change continuation mismatches that clear the plan and re-enter
  ordinary ambiguous MCTS scheduling.

All output is wrapped in a controlled-only result row rather than re-emitting
the staged parent marker directly.  A caller must never use this tool's output
as an R235 physical-game, resource, topology, CUDA, or immutable binding
receipt.  It creates no network request, Kaggle action, process, GPU work, or
stage mutation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.r240_controlled_parent_route_worker/v1"
RESULT_PREFIX = "R240_CONTROLLED_PARENT_ROUTE_RESULT "
CASES = (
    "high_confidence_no_child",
    "continuation_consume",
    "continuation_fingerprint_mismatch",
    "continuation_action_mismatch",
    "continuation_actor_mismatch",
    "continuation_chance_mismatch",
)
CASE_ALIASES = {
    # Retain the initial worker spelling for callers that adopted the draft
    # before the controlled mismatch matrix was expanded.
    "continuation_mismatch": "continuation_fingerprint_mismatch",
}
CONTROLLED_EVIDENCE_CLASS = (
    "controlled_in_memory_parent_route_regression_not_physical_game_"
    "not_preflight_eligible"
)


class ControlledRouteError(RuntimeError):
    """The controlled route did not prove the intended parent behavior."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlledRouteError(message)


def _is_under(path: str | Path, stage: Path) -> bool:
    try:
        Path(path).resolve().relative_to(stage)
    except (OSError, ValueError):
        return False
    return True


def _stage_listing(stage: Path) -> list[dict[str, Any]]:
    """Cheap immutable-tree sentinel that does not read model bytes twice."""

    if not stage.is_dir() or stage.is_symlink():
        raise ControlledRouteError("stage must be an existing physical directory")
    listing: list[dict[str, Any]] = []
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise ControlledRouteError("controlled worker refuses a symlinked stage member")
        if path.is_file():
            stat = path.stat()
            listing.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "mode": int(stat.st_mode & 0o777),
                }
            )
    return listing


def _stage_member_snapshot(stage: Path) -> dict[str, Any]:
    """Content-address the supplied stage once for controlled-run context.

    This is deliberately only a read-only worker-context snapshot.  It is not
    an immutable package receipt, does not replace the builder/binder's exact
    identity checks, and is never interpreted as physical-game evidence.
    """

    if not stage.is_dir() or stage.is_symlink():
        raise ControlledRouteError("stage must be an existing physical directory")
    digest = hashlib.sha256()
    member_count = 0
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise ControlledRouteError("controlled worker refuses a symlinked stage member")
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(stage).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(int(stat.st_mode & 0o777).to_bytes(4, "big"))
        digest.update(int(stat.st_size).to_bytes(16, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        member_count += 1
    return {
        "schema": "poke_bot.controlled_stage_member_snapshot/v1",
        "sha256": "sha256:" + digest.hexdigest(),
        "member_count": member_count,
        "captures_path_mode_size_and_member_bytes": True,
        "captured_before_controlled_parent_calls": True,
        "immutable_preflight_receipt": False,
    }


@contextlib.contextmanager
def _import_exact_stage(stage: Path) -> Iterator[tuple[Any, Any]]:
    """Load the parent and feature module only from the supplied stage.

    The scoped module restoration makes this usable from a test runner as well
    as from its normal dedicated Python worker.  It deliberately does not
    import the workspace parent entrypoint as a fallback.
    """

    stage = stage.expanduser().resolve()
    if not stage.is_dir() or stage.is_symlink():
        raise ControlledRouteError("stage must be an existing physical directory")
    if not (stage / "main.py").is_file() or (stage / "main.py").is_symlink():
        raise ControlledRouteError("stage is missing a physical main.py")

    original_cwd = Path.cwd()
    original_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    environment_keys = (
        "PYTHONDONTWRITEBYTECODE",
        "CG_LIB_PATH",
        # The staged parent applies these Phase-1 caps before direct policy
        # resolution.  Restore them on scope exit so importing this helper
        # from a test process cannot change that process's later work.
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    old_environment = {key: os.environ.get(key) for key in environment_keys}
    saved_modules = {
        name: sys.modules.pop(name)
        for name in tuple(sys.modules)
        if name == "main" or name == "poke_bot" or name.startswith("poke_bot.")
    }
    try:
        sys.dont_write_bytecode = True
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["CG_LIB_PATH"] = str(stage)
        os.chdir(stage)
        sys.path.insert(0, str(stage))
        importlib.invalidate_caches()
        main = importlib.import_module("main")
        features = importlib.import_module("poke_bot.features")
        for label, module in (("main", main), ("poke_bot.features", features)):
            location = getattr(module, "__file__", None)
            if not isinstance(location, str) or not _is_under(location, stage):
                raise ControlledRouteError(
                    f"sealed stage import for {label} resolved outside the stage"
                )
        yield main, features
    finally:
        for name, module in tuple(sys.modules.items()):
            location = getattr(module, "__file__", None)
            if name == "main" or (
                isinstance(location, str)
                and name.startswith("poke_bot")
                and _is_under(location, stage)
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = original_path
        os.chdir(original_cwd)
        sys.dont_write_bytecode = original_dont_write_bytecode
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _ControlledPolicy:
    """Minimal target-emitting surface consumed by the staged parent only."""

    def __init__(self) -> None:
        self.board_history: list[object] = []
        self.previous_action_history: list[object] = []
        self._previous_action_token: object = None
        self.targets: list[dict[str, Any]] = []
        self.collect_targets = False

    @staticmethod
    def _history_context_limit() -> int:
        return 16


class _ControlledDirect:
    """In-memory frozen-policy adapter with one valid factorized target."""

    def __init__(self, *, features: Any, probability: float) -> None:
        self._features = features
        self.probability = float(probability)
        self.policy = _ControlledPolicy()
        self.branch_calls = 0
        self.deck_calls = 0

    def _turn_order_choice(self, _observation: Mapping[str, Any]) -> None:
        return None

    def _ensure_runtime(self) -> tuple[list[int], object, _ControlledPolicy]:
        # An opaque model object intentionally causes only the parent's
        # diagnostic CUDA observation to be incomplete; it never starts a
        # model, a CUDA context, a simulator, or a child process.
        return [741] * 60, object(), self.policy

    def agent(self, observation: Mapping[str, Any]) -> list[int]:
        if observation.get("select") is None:
            self.deck_calls += 1
            return [741] * 60

        self.branch_calls += 1
        self.policy.board_history.append(("controlled-board", self.branch_calls))
        self.policy.previous_action_history.append(self.policy._previous_action_token)
        limit = self.policy._history_context_limit()
        self.policy.board_history = self.policy.board_history[-limit:]
        self.policy.previous_action_history = self.policy.previous_action_history[-limit:]
        self.policy._previous_action_token = ("controlled-direct", [0])

        if self.policy.collect_targets:
            candidates = [
                list(action)
                for action in self._features.factorized_action_candidates(
                    dict(observation), []
                )
            ]
            _require(candidates == [[0], [1]], "controlled factorized order drifted")
            probability = self.probability
            _require(0.0 <= probability <= 1.0, "controlled probability is invalid")
            self.policy.targets.append(
                {
                    "observation": dict(observation),
                    "action": [0],
                    "factorized_stages": [
                        {
                            "action_combos": candidates,
                            "policy": [probability, 1.0 - probability],
                            "selected_index": 0,
                        }
                    ],
                    "diagnostics": {
                        "target_source": "history_policy",
                        "trusted": True,
                        "history_length": len(self.policy.board_history),
                    },
                }
            )
        return [0]


def _normal_two_lane_receipt(
    *, principal_variation: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A controlled receipt that the staged parent must independently accept.

    It deliberately carries no terminal-win claim and exposes the two
    handle-scoped SearchId composites expected by the parent.  It is a
    controlled protocol input, never simulator telemetry.
    """

    return {
        "selected_action": [1],
        "mcts_action_authority": True,
        "mode": "shared_tree_mcts",
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_release_calls": 2,
        "search_end_calls": 2,
        "search_step_calls": 2,
        "per_lane_depth": [1, 1],
        "completed_backups": 2,
        "root_visits": 2,
        "per_lane_handle_identities": ["controlled-lane-0", "controlled-lane-1"],
        "per_lane_search_id_chains": [[0], [0]],
        "per_lane_first_search_ids": [0, 0],
        "distinct_search_begin_composite_count": 2,
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": "controlled-lane-0", "first_search_id": 0},
            {"lane_id": 1, "handle_identity": "controlled-lane-1", "first_search_id": 0},
        ],
        "microbatch_sizes": [2],
        "max_simulator_calls_in_flight": 2,
        "outstanding_virtual_loss": 0,
        "stop_reason": "tree_exhausted",
        "minimum_backups_before_stability": 8,
        "stable_root_leader_observations_required": 3,
        "maximum_backups_per_decision": 32,
        "observed_stable_root_leader_observations": 0,
        "root_seat": 0,
        "principal_variation": [dict(entry) for entry in principal_variation],
        "terminal_win_proof": None,
        "proven_deterministic_terminal_win_this_turn": False,
    }


class _ControlledBroker:
    """No-process broker adapter whose receipt still crosses parent validation."""

    def __init__(self, *, receipt_factory: Callable[[], Mapping[str, Any]], **kwargs: Any) -> None:
        self.receipt_factory = receipt_factory
        self.constructor_kwargs = dict(kwargs)
        self.begin_game_calls = 0
        self.select_calls: list[tuple[dict[str, Any], list[int]]] = []
        self.note_calls: list[tuple[dict[str, Any], list[int]]] = []
        self.close_calls = 0
        self.degraded = False
        self.disabled = False
        self.last_fault: dict[str, Any] | None = None
        # The staged parent uses this only to distinguish a history-only note
        # to a prior child from deferred local journal state.
        self.has_live_child = True

    def begin_game(self, *, start_child: bool = True) -> None:
        self.begin_game_calls += 1
        _require(start_child is False, "parent unexpectedly requested eager child start")

    def select(
        self, observation: Mapping[str, Any], direct_action: Sequence[int]
    ) -> tuple[list[int], dict[str, Any], None]:
        self.select_calls.append((dict(observation), list(direct_action)))
        _require(list(direct_action) == [0], "controlled direct action drifted")
        return [1], dict(self.receipt_factory()), None

    def note_direct_action(
        self, observation: Mapping[str, Any], action: Sequence[int]
    ) -> None:
        self.note_calls.append((dict(observation), list(action)))

    def marker_payload(self) -> dict[str, Any]:
        return {
            "schema": "poke_bot.r228_kaggle_subprocess_broker/v1",
            "controlled_adapter": True,
            "disabled": self.disabled,
            "degraded": self.degraded,
            "child_pid": None,
            "last_fault": None,
            "progress_by_lane": {},
            "select_call_count": len(self.select_calls),
            "note_direct_action_call_count": len(self.note_calls),
        }

    def close(self) -> None:
        self.close_calls += 1


class _ControlledBrokerFactory:
    def __init__(self, receipt_factory: Callable[[], Mapping[str, Any]]) -> None:
        self._receipt_factory = receipt_factory
        self.instances: list[_ControlledBroker] = []

    def __call__(self, **kwargs: Any) -> _ControlledBroker:
        instance = _ControlledBroker(receipt_factory=self._receipt_factory, **kwargs)
        self.instances.append(instance)
        return instance


@contextlib.contextmanager
def _controlled_parent_adapters(
    *, main: Any, features: Any, direct: _ControlledDirect, broker_factory: _ControlledBrokerFactory
) -> Iterator[None]:
    """Install only in-memory controlled seams and restore them afterwards."""

    originals = {
        "direct": main._direct,
        "broker": main.IsolatedR228SearchBroker,
        "enumerate": features.enumerate_action_combos,
        "factorized": features.factorized_action_candidates,
        "tokens": features.build_option_tokens,
    }
    try:
        main._direct = lambda: direct
        main.IsolatedR228SearchBroker = broker_factory
        features.enumerate_action_combos = (
            lambda _obs, *, max_combos: [[0], [1]]
            if max_combos == 65_536
            else (_ for _ in ()).throw(
                ControlledRouteError("parent legal enumeration cap drifted")
            )
        )
        features.factorized_action_candidates = lambda _obs, _prefix: [[0], [1]]
        features.build_option_tokens = (
            lambda _obs, actions: ("controlled-parent-history-token", [list(row) for row in actions])
        )
        yield
    finally:
        main._direct = originals["direct"]
        main.IsolatedR228SearchBroker = originals["broker"]
        features.enumerate_action_combos = originals["enumerate"]
        features.factorized_action_candidates = originals["factorized"]
        features.build_option_tokens = originals["tokens"]


def _deck_observation() -> dict[str, Any]:
    return {"current": None, "select": None}


def _branch_observation() -> dict[str, Any]:
    return {
        "current": {"yourIndex": 0},
        "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
    }


def _decision_markers(raw: str, *, prefix: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    marker_prefix = prefix + " "
    for line in raw.splitlines():
        if not line.startswith(marker_prefix):
            continue
        try:
            payload = json.loads(line[len(marker_prefix) :])
        except json.JSONDecodeError as exc:
            raise ControlledRouteError("staged parent emitted a malformed decision marker") from exc
        if not isinstance(payload, dict):
            raise ControlledRouteError("staged parent decision marker is not an object")
        markers.append(payload)
    return markers


def _common_route_setup(
    stage: Path,
    *, probability: float,
    plan_factory: Callable[[Any, Mapping[str, Any]], Callable[[], Mapping[str, Any]]],
    exercise: Callable[
        [
            Any,
            _ControlledDirect,
            _ControlledBrokerFactory,
            Mapping[str, Any],
            Callable[[str, Mapping[str, Any], bool], list[int]],
        ],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    """Run one controlled case while retaining only wrapper-level evidence."""

    stage_before = _stage_listing(stage)
    captured = io.StringIO()
    try:
        with _import_exact_stage(stage) as (main, features):
            observation = _branch_observation()
            direct = _ControlledDirect(features=features, probability=probability)
            receipt_factory = plan_factory(main, observation)
            factory = _ControlledBrokerFactory(receipt_factory)
            agent_calls: list[dict[str, Any]] = []
            with _controlled_parent_adapters(
                main=main,
                features=features,
                direct=direct,
                broker_factory=factory,
            ), contextlib.redirect_stdout(captured):
                def call_agent(
                    label: str, event_observation: Mapping[str, Any], is_decision: bool
                ) -> list[int]:
                    # Capture the staged parent's own canonical fingerprint
                    # immediately before its matching ``agent`` call.  This is
                    # deliberately a fingerprint of a controlled in-memory
                    # input, not physical-game telemetry; retaining it lets a
                    # higher-level controlled-only mapper bind a raw parent
                    # marker to the exact root it exercised.
                    controlled_root_observation_fingerprint: str | None = None
                    if is_decision:
                        try:
                            fingerprint = main._canonical_observation_fingerprint(
                                dict(event_observation)
                            )
                        except Exception as exc:
                            raise ControlledRouteError(
                                "staged main could not fingerprint controlled root"
                            ) from exc
                        _require(
                            isinstance(fingerprint, str) and bool(fingerprint),
                            "staged main returned no controlled root fingerprint",
                        )
                        controlled_root_observation_fingerprint = fingerprint
                    before_count = len(
                        _decision_markers(
                            captured.getvalue(), prefix=main.DECISION_PREFIX
                        )
                    )
                    call_started = time.monotonic()
                    action = main.agent(dict(event_observation))
                    elapsed_seconds = max(0.0, time.monotonic() - call_started)
                    after_count = len(
                        _decision_markers(
                            captured.getvalue(), prefix=main.DECISION_PREFIX
                        )
                    )
                    agent_calls.append(
                        {
                            "label": label,
                            "is_decision": is_decision,
                            "elapsed_seconds": elapsed_seconds,
                            "selected_action": list(action),
                            "decision_marker_indices": list(
                                range(before_count, after_count)
                            ),
                            "timing_scope": (
                                "controlled_parent_main_agent_call_only_"
                                "not_physical_latency_or_resource_evidence"
                            ),
                            "controlled_root_observation_fingerprint": (
                                controlled_root_observation_fingerprint
                            ),
                            "controlled_root_observation_fingerprint_source": (
                                "staged_main._canonical_observation_fingerprint"
                                if is_decision
                                else None
                            ),
                        }
                    )
                    return action

                deck = call_agent("deck_boundary", _deck_observation(), False)
                _require(deck == [741] * 60, "staged parent did not return controlled deck")
                checks = exercise(main, direct, factory, observation, call_agent)
            decisions = _decision_markers(captured.getvalue(), prefix=main.DECISION_PREFIX)
            for call in agent_calls:
                if call["is_decision"]:
                    _require(
                        len(call["decision_marker_indices"]) == 1,
                        "each controlled decision call must emit exactly one parent marker",
                    )
            return {
                "status": "passed",
                "stage_import": {
                    "main": str(Path(main.__file__).resolve()),
                    "features": str(Path(features.__file__).resolve()),
                    "stock_library_identity_check_executed_by_staged_parent": True,
                },
                "checks": checks,
                "parent_decision_markers": decisions,
                "main_agent_calls": agent_calls,
                "adapter_calls": {
                    "direct_branch_calls": direct.branch_calls,
                    "direct_deck_calls": direct.deck_calls,
                    "broker_instance_count": len(factory.instances),
                    "broker_select_call_count": sum(
                        len(instance.select_calls) for instance in factory.instances
                    ),
                    "broker_note_direct_action_call_count": sum(
                        len(instance.note_calls) for instance in factory.instances
                    ),
                },
            }
    finally:
        # The listing also detects accidental bytecode output even though the
        # import scope disables it.  Content identity is owned by the package
        # builder/binder; this route helper deliberately avoids a second model
        # hash over a potentially large sealed member.
        stage_after = _stage_listing(stage)
        # ``return`` inside the try evaluates before finally, so attach these
        # fields through the local exception-free result below is impossible.
        # Raise only on drift; the caller builds the immutable report fields.
        if stage_after != stage_before:
            raise ControlledRouteError("controlled worker mutated the sealed stage")


def _run_high_confidence(stage: Path) -> dict[str, Any]:
    def plan_factory(_main: Any, _observation: Mapping[str, Any]) -> Callable[[], Mapping[str, Any]]:
        return lambda: _normal_two_lane_receipt(principal_variation=[])

    def exercise(
        main: Any,
        direct: _ControlledDirect,
        factory: _ControlledBrokerFactory,
        observation: Mapping[str, Any],
        call_agent: Callable[[str, Mapping[str, Any], bool], list[int]],
    ) -> dict[str, Any]:
        selected = call_agent("high_confidence_direct", observation, True)
        _require(selected == [0], "high-confidence route did not return direct action")
        _require(factory.instances == [], "high-confidence route constructed a broker")
        _require(direct.branch_calls == 1, "high-confidence route called direct policy unexpectedly")
        return {
            "route_label": "r242_high_confidence_frozen_direct_no_child",
            "selected_action": selected,
            "broker_constructed": False,
        }

    result = _common_route_setup(
        stage,
        probability=0.80,
        plan_factory=plan_factory,
        exercise=exercise,
    )
    markers = result["parent_decision_markers"]
    _require(len(markers) == 1, "high-confidence route did not emit exactly one marker")
    marker = markers[0]
    _require(marker.get("mode") == "high_confidence_frozen_direct", "high-confidence marker mode drifted")
    _require(marker.get("mcts_child_started_for_this_decision") is False, "high-confidence marker started child")
    _require(marker.get("mcts_select_call_count") == 0, "high-confidence marker selected MCTS")
    _require(marker.get("history_only_existing_child_journal_count") == 0, "unexpected high-confidence journal IPC")
    return result


def _run_continuation_consume(stage: Path) -> dict[str, Any]:
    plan_state: dict[str, list[dict[str, Any]]] = {"entries": []}

    def plan_factory(main: Any, observation: Mapping[str, Any]) -> Callable[[], Mapping[str, Any]]:
        fingerprint = main._canonical_observation_fingerprint(observation)
        _require(isinstance(fingerprint, str) and bool(fingerprint), "stage cannot fingerprint controlled root")
        plan_state["entries"] = [{"observation_fingerprint": fingerprint, "action": [1]}]
        return lambda: _normal_two_lane_receipt(principal_variation=plan_state["entries"])

    def exercise(
        main: Any,
        _direct: _ControlledDirect,
        factory: _ControlledBrokerFactory,
        observation: Mapping[str, Any],
        call_agent: Callable[[str, Mapping[str, Any], bool], list[int]],
    ) -> dict[str, Any]:
        first = call_agent("ambiguous_mcts_plan_extraction", observation, True)
        _require(first == [1], "controlled searched route did not return broker action")
        plan_state["entries"] = []
        second = call_agent("deterministic_continuation_consume", observation, True)
        _require(second == [1], "deterministic continuation did not consume planned action")
        _require(len(factory.instances) == 1, "continuation route replaced its broker")
        broker = factory.instances[0]
        _require(len(broker.select_calls) == 1, "continuation started a second MCTS select")
        _require(
            broker.note_calls == [(dict(observation), [1])],
            "continuation did not issue exactly one history-only note_direct_action",
        )
        return {
            "route_label": "r240_deterministic_continuation_valid_consume",
            "first_selected_action": first,
            "second_selected_action": second,
            "fresh_mcts_selects_after_plan": 0,
            "history_only_note_direct_action_calls": 1,
        }

    result = _common_route_setup(
        stage,
        probability=0.50,
        plan_factory=plan_factory,
        exercise=exercise,
    )
    markers = result["parent_decision_markers"]
    _require(len(markers) == 2, "continuation consume did not emit two parent markers")
    _require(markers[0].get("mode") == "shared_tree_mcts", "initial continuation route was not ambiguous MCTS")
    marker = markers[1]
    _require(marker.get("mode") == "deterministic_mcts_continuation", "valid continuation was not consumed")
    _require(marker.get("mcts_child_started_for_this_decision") is False, "continuation started a child")
    _require(marker.get("mcts_select_call_count") == 0, "continuation selected MCTS")
    _require(marker.get("history_only_existing_child_journal_count") == 1, "continuation journal count drifted")
    _require(marker.get("history_rewritten_to_actual_action") is True, "continuation did not rewrite history")
    return result


def _mismatch_followup_observation(kind: str) -> dict[str, Any]:
    """Return an explicit controlled input for one parent clear-plan guard."""

    observation = _branch_observation()
    if kind == "actor":
        observation["current"] = {"yourIndex": 1}
    elif kind == "chance":
        # This is not a stock chance simulation.  It is an explicit changed
        # canonical parent input, with a controlled label, so the parent must
        # invalidate a plan extracted for the prior root fingerprint.
        observation["controlled_chance_boundary"] = {
            "classification": "controlled_observed_chance_boundary",
            "not_stock_simulation": True,
        }
    return observation


def _run_continuation_mismatch(stage: Path, *, kind: str) -> dict[str, Any]:
    """Prove one actual parent guard clears a controlled continuation plan."""

    if kind not in {"fingerprint", "action", "actor", "chance"}:
        raise ControlledRouteError(f"unknown continuation mismatch kind: {kind!r}")
    plan_state: dict[str, list[dict[str, Any]]] = {"entries": []}
    preconditions: dict[str, Any] = {}

    def plan_factory(main: Any, observation: Mapping[str, Any]) -> Callable[[], Mapping[str, Any]]:
        root_fingerprint = main._canonical_observation_fingerprint(observation)
        _require(
            isinstance(root_fingerprint, str) and bool(root_fingerprint),
            "stage cannot fingerprint controlled root",
        )
        planned_action = [9] if kind == "action" else [1]
        planned_fingerprint = (
            "sha256:controlled-stale-root"
            if kind == "fingerprint"
            else root_fingerprint
        )
        plan_state["entries"] = [
            {
                "observation_fingerprint": planned_fingerprint,
                "action": planned_action,
            }
        ]
        followup = _mismatch_followup_observation(kind)
        followup_fingerprint = main._canonical_observation_fingerprint(followup)
        _require(
            isinstance(followup_fingerprint, str) and bool(followup_fingerprint),
            "stage cannot fingerprint controlled mismatch input",
        )
        preconditions.update(
            {
                "controlled_mismatch_kind": kind,
                "planned_action": planned_action,
                "planned_fingerprint_equals_initial_root": planned_fingerprint
                == root_fingerprint,
                "followup_fingerprint_equals_initial_root": followup_fingerprint
                == root_fingerprint,
                "initial_actor_seat": 0,
                "followup_actor_seat": followup["current"]["yourIndex"],
                "controlled_chance_boundary_injected": kind == "chance",
            }
        )
        return lambda: _normal_two_lane_receipt(
            principal_variation=plan_state["entries"]
        )

    def exercise(
        main: Any,
        _direct: _ControlledDirect,
        factory: _ControlledBrokerFactory,
        observation: Mapping[str, Any],
        call_agent: Callable[[str, Mapping[str, Any], bool], list[int]],
    ) -> dict[str, Any]:
        first = call_agent("ambiguous_mcts_plan_extraction", observation, True)
        _require(first == [1], "initial ambiguous route did not return broker action")
        # The first controlled plan remains stored.  The next normal controlled
        # reply contains no plan, so a nonempty parent plan here would prove
        # stale state was retained rather than cleared.
        plan_state["entries"] = []
        followup = _mismatch_followup_observation(kind)
        second = call_agent(
            f"deterministic_continuation_{kind}_mismatch", followup, True
        )
        _require(second == [1], "mismatch route did not re-enter normal MCTS")
        _require(len(factory.instances) == 1, "mismatch route replaced its broker")
        broker = factory.instances[0]
        _require(
            len(broker.select_calls) == 2,
            "mismatched plan did not re-enter MCTS select",
        )
        _require(broker.note_calls == [], "mismatched plan was consumed or journaled")
        _require(main._GAME_PRINCIPAL_VARIATION == [], "mismatched plan was not cleared")
        return {
            "route_label": f"r240_deterministic_continuation_{kind}_mismatch_clears_plan",
            "first_selected_action": first,
            "second_selected_action": second,
            "fresh_mcts_selects_after_mismatch": 1,
            "history_only_note_direct_action_calls": 0,
            "controlled_guard_preconditions": dict(preconditions),
            "entire_plan_cleared_before_second_normal_mcts_select": True,
        }

    result = _common_route_setup(
        stage,
        probability=0.50,
        plan_factory=plan_factory,
        exercise=exercise,
    )
    markers = result["parent_decision_markers"]
    _require(len(markers) == 2, "continuation mismatch did not emit two parent markers")
    _require(
        [marker.get("mode") for marker in markers]
        == ["shared_tree_mcts", "shared_tree_mcts"],
        "mismatched continuation did not return to ordinary ambiguous MCTS",
    )
    return result


def _run_continuation_fingerprint_mismatch(stage: Path) -> dict[str, Any]:
    return _run_continuation_mismatch(stage, kind="fingerprint")


def _run_continuation_action_mismatch(stage: Path) -> dict[str, Any]:
    return _run_continuation_mismatch(stage, kind="action")


def _run_continuation_actor_mismatch(stage: Path) -> dict[str, Any]:
    return _run_continuation_mismatch(stage, kind="actor")


def _run_continuation_chance_mismatch(stage: Path) -> dict[str, Any]:
    return _run_continuation_mismatch(stage, kind="chance")


def run_case(
    *,
    stage: Path,
    case: str,
    stage_member_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one route and return an explicitly controlled-only result."""

    runners: dict[str, Callable[[Path], dict[str, Any]]] = {
        "high_confidence_no_child": _run_high_confidence,
        "continuation_consume": _run_continuation_consume,
        "continuation_fingerprint_mismatch": _run_continuation_fingerprint_mismatch,
        "continuation_action_mismatch": _run_continuation_action_mismatch,
        "continuation_actor_mismatch": _run_continuation_actor_mismatch,
        "continuation_chance_mismatch": _run_continuation_chance_mismatch,
    }
    case = CASE_ALIASES.get(case, case)
    if case not in runners:
        raise ControlledRouteError(f"unsupported controlled route: {case!r}")
    stage = stage.expanduser().resolve()
    started = time.monotonic()
    listing_before = _stage_listing(stage)
    try:
        snapshot = (
            dict(stage_member_snapshot)
            if stage_member_snapshot is not None
            else _stage_member_snapshot(stage)
        )
        result = runners[case](stage)
        listing_after = _stage_listing(stage)
        _require(listing_after == listing_before, "controlled route changed the sealed stage")
        return {
            "schema": SCHEMA,
            "status": "passed",
            "controlled": True,
            "evidence_kind": "controlled_parent_route",
            "controlled_parent_route": True,
            "evidence_class": CONTROLLED_EVIDENCE_CLASS,
            "route_case": case,
            "network_accessed": False,
            "kaggle_api_called": False,
            "kaggle_upload_used": False,
            "gpu_used": False,
            "simulator_started": False,
            "model_loaded": False,
            "stage_mutation_check": {"unchanged": True},
            "stage_member_snapshot": snapshot,
            "result": result,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
        }
    except Exception as exc:  # noqa: BLE001 - the result must preserve its controlled failure
        return {
            "schema": SCHEMA,
            "status": "failed_closed",
            "controlled": True,
            "evidence_kind": "controlled_parent_route",
            "controlled_parent_route": True,
            "evidence_class": CONTROLLED_EVIDENCE_CLASS,
            "route_case": case,
            "network_accessed": False,
            "kaggle_api_called": False,
            "kaggle_upload_used": False,
            "gpu_used": False,
            "simulator_started": False,
            "model_loaded": False,
            "stage_member_snapshot": (
                None if stage_member_snapshot is None else dict(stage_member_snapshot)
            ),
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "elapsed_seconds": max(0.0, time.monotonic() - started),
        }


def _marker_for_agent_call(
    result: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one captured staged-parent marker to its timed call record."""

    inner = result.get("result")
    if not isinstance(inner, Mapping):
        raise ControlledRouteError("controlled case omitted its route result")
    calls = inner.get("main_agent_calls")
    markers = inner.get("parent_decision_markers")
    if not isinstance(calls, list) or not isinstance(markers, list):
        raise ControlledRouteError("controlled case omitted parent markers or timings")
    matching = [call for call in calls if isinstance(call, Mapping) and call.get("label") == label]
    if len(matching) != 1:
        raise ControlledRouteError(f"controlled case lacks one timed {label!r} call")
    call = dict(matching[0])
    indices = call.get("decision_marker_indices")
    if not isinstance(indices, list) or len(indices) != 1 or not isinstance(indices[0], int):
        raise ControlledRouteError(f"controlled call {label!r} does not bind one marker")
    index = indices[0]
    if not 0 <= index < len(markers) or not isinstance(markers[index], Mapping):
        raise ControlledRouteError(f"controlled call {label!r} points outside marker telemetry")
    return dict(markers[index]), call


def _controlled_root_fingerprint_for_agent_call(
    call: Mapping[str, Any], *, label: str
) -> str:
    """Return the staged-main fingerprint recorded before one controlled call."""

    fingerprint = call.get("controlled_root_observation_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ControlledRouteError(
            f"controlled call {label!r} lacks a staged-main root fingerprint"
        )
    if call.get("controlled_root_observation_fingerprint_source") != (
        "staged_main._canonical_observation_fingerprint"
    ):
        raise ControlledRouteError(
            f"controlled call {label!r} root fingerprint did not come from staged main"
        )
    return fingerprint


def _case_result(results: Sequence[Mapping[str, Any]], case: str) -> Mapping[str, Any]:
    matching = [row for row in results if row.get("route_case") == case]
    if len(matching) != 1:
        raise ControlledRouteError(f"controlled parent matrix lacks case {case!r}")
    if matching[0].get("status") != "passed":
        raise ControlledRouteError(f"controlled parent matrix case {case!r} did not pass")
    return matching[0]


def _controlled_normalized_parent_route_evidence(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project measured controlled markers into the outer-runner handoff map.

    Names intentionally resemble the consuming normalizer's field vocabulary,
    but the surrounding object is unambiguously controlled-only.  In
    particular, its broker-select rows are in-memory protocol adapters, not
    stock simulator, child-process, resource, CUDA, terminal-win, or full-game
    evidence.
    """

    try:
        high_result = _case_result(results, "high_confidence_no_child")
        consume_result = _case_result(results, "continuation_consume")
        mismatch_results = {
            kind: _case_result(results, f"continuation_{kind}_mismatch")
            for kind in ("chance", "fingerprint", "action", "actor")
        }
        high_marker, high_call = _marker_for_agent_call(
            high_result, label="high_confidence_direct"
        )
        consume_search_marker, consume_search_call = _marker_for_agent_call(
            consume_result, label="ambiguous_mcts_plan_extraction"
        )
        consume_marker, consume_call = _marker_for_agent_call(
            consume_result, label="deterministic_continuation_consume"
        )
        high_root_fingerprint = _controlled_root_fingerprint_for_agent_call(
            high_call, label="high_confidence_direct"
        )
        consume_search_root_fingerprint = _controlled_root_fingerprint_for_agent_call(
            consume_search_call, label="ambiguous_mcts_plan_extraction"
        )
        _controlled_root_fingerprint_for_agent_call(
            consume_call, label="deterministic_continuation_consume"
        )

        if high_marker.get("mode") != "high_confidence_frozen_direct":
            raise ControlledRouteError("controlled high route did not emit direct mode")
        if consume_search_marker.get("mode") != "shared_tree_mcts":
            raise ControlledRouteError("controlled plan extraction did not emit shared MCTS")
        if consume_marker.get("mode") != "deterministic_mcts_continuation":
            raise ControlledRouteError("controlled continuation did not emit consume mode")

        probabilities = high_marker.get("selected_factorized_stage_probabilities")
        if not isinstance(probabilities, list) or not probabilities:
            raise ControlledRouteError("controlled high route omitted stage probabilities")
        direct_action = high_marker.get("direct_action")
        if not isinstance(direct_action, list):
            raise ControlledRouteError("controlled high route omitted direct action")
        high = {
            "controlled_only": True,
            "nonphysical": True,
            "selected_factorized_stage_probabilities": list(probabilities),
            "selected_factorized_stage_probability_threshold": high_marker.get(
                "selected_factorized_stage_probability_threshold"
            ),
            "all_selected_factorized_stages_meet_threshold": high_marker.get(
                "all_selected_factorized_stages_meet_threshold"
            ),
            "mode": high_marker.get("mode"),
            # The staged parent called and independently checked the temporary
            # direct-policy target before this branch.  Its direct adapter is
            # controlled, so this remains a parent-route fact only.
            "direct_action_precomputed_and_validated": True,
            "mcts_child_started_for_this_decision": high_marker.get(
                "mcts_child_started_for_this_decision"
            ),
            "mcts_select_call_count": high_marker.get("mcts_select_call_count"),
            # A high-confidence branch constructed no controlled broker.  The
            # staged marker proves no child/select; these derived zeroes are
            # bounded to the controlled adapter scope.
            "mcts_search_call_count": 0,
            "mcts_model_call_count": 0,
            "mcts_simulator_call_count": 0,
            "history_only_existing_child_journal_count": high_marker.get(
                "history_only_existing_child_journal_count"
            ),
            "degraded": high_marker.get("degraded"),
            "selected_action": high_marker.get("selected_action"),
            "direct_action": direct_action,
            "parent_action_elapsed_seconds": high_call.get("elapsed_seconds"),
        }

        events: list[dict[str, Any]] = []

        def append_event(
            *,
            result: Mapping[str, Any],
            label: str,
            normalized_mode: str,
        ) -> None:
            marker, call = _marker_for_agent_call(result, label=label)
            root_fingerprint = _controlled_root_fingerprint_for_agent_call(
                call, label=label
            )
            events.append(
                {
                    "controlled_only": True,
                    "route_case": result.get("route_case"),
                    "agent_call_label": label,
                    "mode": normalized_mode,
                    "source_parent_mode": marker.get("mode"),
                    "selected_action": marker.get("selected_action"),
                    "direct_action": marker.get("direct_action"),
                    "mcts_child_started_for_this_decision": marker.get(
                        "mcts_child_started_for_this_decision"
                    ),
                    "mcts_select_call_count": marker.get("mcts_select_call_count"),
                    "history_only_existing_child_journal_count": marker.get(
                        "history_only_existing_child_journal_count"
                    ),
                    "degraded": marker.get("degraded"),
                    "parent_action_elapsed_seconds": call.get("elapsed_seconds"),
                    "controlled_child_search_elapsed_seconds": 0.0,
                    # This exact value was obtained through the imported
                    # staged main immediately before this controlled parent
                    # call.  It is not a fingerprint of a physical game.
                    "controlled_root_observation_fingerprint": root_fingerprint,
                    "controlled_root_observation_fingerprint_source": (
                        "staged_main._canonical_observation_fingerprint"
                    ),
                }
            )

        append_event(
            result=high_result,
            label="high_confidence_direct",
            normalized_mode="high_confidence_frozen_direct",
        )
        append_event(
            result=consume_result,
            label="ambiguous_mcts_plan_extraction",
            normalized_mode="new_adaptive_two_lane_mcts",
        )
        append_event(
            result=consume_result,
            label="deterministic_continuation_consume",
            normalized_mode="cached_deterministic_continuation",
        )
        for kind, result in mismatch_results.items():
            append_event(
                result=result,
                label="ambiguous_mcts_plan_extraction",
                normalized_mode="new_adaptive_two_lane_mcts",
            )
            append_event(
                result=result,
                label=f"deterministic_continuation_{kind}_mismatch",
                normalized_mode="new_adaptive_two_lane_mcts",
            )

        mismatch_clears: dict[str, bool] = {}
        for kind, result in mismatch_results.items():
            inner = result.get("result")
            if not isinstance(inner, Mapping) or not isinstance(inner.get("checks"), Mapping):
                raise ControlledRouteError(f"controlled {kind} case lacks clear-plan checks")
            checks = inner["checks"]
            mismatch_clears[kind] = (
                checks.get("entire_plan_cleared_before_second_normal_mcts_select")
                is True
                and checks.get("fresh_mcts_selects_after_mismatch") == 1
                and checks.get("history_only_note_direct_action_calls") == 0
            )
        if not all(mismatch_clears.values()):
            raise ControlledRouteError("a controlled mismatch did not clear the entire plan")

        continuation_fingerprint = consume_marker.get(
            "continuation_observation_fingerprint"
        )
        planned_action = consume_marker.get("planned_action")
        if not isinstance(continuation_fingerprint, str) or not isinstance(planned_action, list):
            raise ControlledRouteError("controlled continuation marker omitted exact consume binding")
        continuation_plan = {
            "controlled_only": True,
            "plan_id": "controlled-parent-route-plan-001",
            "actual_turn_id": "controlled-parent-route-turn-001",
            "extracted_from_mode": "controlled_shared_tree_mcts_receipt",
            "exact_fingerprint_proven_by_controlled_parent_receipt": True,
            "two_lane_agreed_backed_leader_asserted_by_controlled_receipt": True,
            "no_chance_boundary_or_opponent_transition_asserted_by_controlled_receipt": True,
            "steps": [
                {
                    "canonical_observation_fingerprint": continuation_fingerprint,
                    "planned_action": list(planned_action),
                    "consumed_exactly_once_by_staged_parent": True,
                    "parent_action_elapsed_seconds": consume_call.get("elapsed_seconds"),
                }
            ],
        }
        precomputed_and_history_retained = (
            consume_marker.get("history_rewritten_to_actual_action") is True
            and isinstance(consume_marker.get("direct_action"), list)
            and isinstance(consume_marker.get("selected_action"), list)
        )
        parent_seconds = sum(
            float(event["parent_action_elapsed_seconds"])
            for event in events
            if isinstance(event.get("parent_action_elapsed_seconds"), (int, float))
            and not isinstance(event.get("parent_action_elapsed_seconds"), bool)
        )
        new_search_count = sum(
            event["mode"] == "new_adaptive_two_lane_mcts" for event in events
        )
        cached_count = sum(
            event["mode"] == "cached_deterministic_continuation" for event in events
        )
        high_count = sum(
            event["mode"] == "high_confidence_frozen_direct" for event in events
        )
        return {
            "controlled_only": True,
            "nonphysical": True,
            "not_r240_final_schema": True,
            "synthetic_high_confidence_direct": high,
            "full_game_cumulative": {
                "controlled_only": True,
                "nonphysical": True,
                # Preserve the actual marker emitted by the staged parent for
                # the controlled initial MCTS call.  In particular, an outer
                # mapper can inspect the parent-validated two-lane fields
                # without converting the in-memory adapter into a stock child
                # or physical simulator assertion.
                "controlled_plan_extraction_marker": dict(consume_search_marker),
                "controlled_plan_extraction_marker_is_verbatim_staged_parent_marker": True,
                "controlled_plan_extraction_marker_scope": (
                    "controlled_in_memory_adapter_marker_not_physical_child_"
                    "simulator_resource_or_terminal_evidence"
                ),
                "controlled_plan_extraction_root_observation_fingerprint": (
                    consume_search_root_fingerprint
                ),
                "controlled_high_confidence_root_observation_fingerprint": (
                    high_root_fingerprint
                ),
                "cumulative_parent_wall_seconds": parent_seconds,
                # Controlled adapters deliberately launch no child process;
                # zero means no physical child-search timing was observed.
                "cumulative_child_search_seconds": 0.0,
                "new_mcts_search_count": int(new_search_count),
                "cached_deterministic_continuation_count": int(cached_count),
                "high_confidence_frozen_direct_count": int(high_count),
                "deterministic_continuation_plans": [continuation_plan],
                "decision_events": events,
                "deterministic_continuation_regression": {
                    "chance_disagreement_clears_entire_plan": mismatch_clears["chance"],
                    "fingerprint_disagreement_clears_entire_plan": mismatch_clears[
                        "fingerprint"
                    ],
                    "action_disagreement_clears_entire_plan": mismatch_clears["action"],
                    "actor_disagreement_clears_entire_plan": mismatch_clears["actor"],
                    "precomputed_direct_action_and_history_correction_retained": (
                        precomputed_and_history_retained
                    ),
                },
                "exact_consumed_continuation_event": {
                    "source_parent_mode": consume_marker.get("mode"),
                    "selected_action": consume_marker.get("selected_action"),
                    "direct_action": consume_marker.get("direct_action"),
                    "history_rewritten_to_actual_action": consume_marker.get(
                        "history_rewritten_to_actual_action"
                    ),
                    "parent_action_elapsed_seconds": consume_call.get("elapsed_seconds"),
                },
            },
        }
    except Exception as exc:  # preserve a controlled-unavailable shape for the outer mapper
        return {
            "controlled_only": True,
            "nonphysical": True,
            "not_r240_final_schema": True,
            "available": False,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--case", choices=(*CASES, *CASE_ALIASES, "all"), default="all")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_cases = (
        CASES
        if args.case == "all"
        else (CASE_ALIASES.get(str(args.case), str(args.case)),)
    )
    # A sealed import is not expected to print, but capture it anyway so this
    # executable itself always produces exactly one typed stdout object.  Any
    # unexpected import-level output makes the wrapper fail closed rather than
    # allowing an unlabelled staged marker to be mistaken for evidence.
    unexpected_stdout = io.StringIO()
    stage_mutation_check: dict[str, Any] = {
        "unchanged": False,
        "scope": "controlled_worker_pre_and_post_all_case_stage_listing",
    }
    with contextlib.redirect_stdout(unexpected_stdout):
        try:
            stage = args.stage.expanduser().resolve()
            stage_listing_before = _stage_listing(stage)
            stage_snapshot = _stage_member_snapshot(stage)
            results = [
                run_case(
                    stage=stage,
                    case=case,
                    stage_member_snapshot=stage_snapshot,
                )
                for case in selected_cases
            ]
            stage_listing_after = _stage_listing(stage)
            _require(
                stage_listing_after == stage_listing_before,
                "controlled worker changed the sealed stage across its full matrix",
            )
            stage_mutation_check = {
                "unchanged": True,
                "scope": "controlled_worker_pre_and_post_all_case_stage_listing",
                "checked_before_and_after_all_cases": True,
            }
        except Exception as exc:  # no raw exception escapes the one-object worker protocol
            stage_snapshot = None
            results = [
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "controlled": True,
                    "evidence_kind": "controlled_parent_route",
                    "controlled_parent_route": True,
                    "evidence_class": CONTROLLED_EVIDENCE_CLASS,
                    "route_case": case,
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
                for case in selected_cases
            ]
    unexpected_text = unexpected_stdout.getvalue()
    normalized = _controlled_normalized_parent_route_evidence(results)
    payload = {
        "schema": SCHEMA,
        "controlled": True,
        "evidence_kind": "controlled_parent_route",
        "controlled_parent_route": True,
        "evidence_class": CONTROLLED_EVIDENCE_CLASS,
        "network_accessed": False,
        "kaggle_api_called": False,
        "kaggle_upload_used": False,
        "gpu_used": False,
        "simulator_started": False,
        "model_loaded": False,
        "status": (
            "passed"
            if (
                not unexpected_text
                and all(row.get("status") == "passed" for row in results)
                and (
                    args.case != "all"
                    or normalized.get("available", True) is not False
                )
            )
            else "failed_closed"
        ),
        "stage_member_snapshot": stage_snapshot,
        "stage_mutation_check": stage_mutation_check,
        "unexpected_worker_stdout": {
            "bytes": len(unexpected_text.encode("utf-8", errors="replace")),
            "present": bool(unexpected_text),
            "tail": unexpected_text[-4096:],
        },
        "route_results": results,
        "normalized_parent_route_evidence": normalized,
    }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - package-external worker entrypoint.
    raise SystemExit(main())
