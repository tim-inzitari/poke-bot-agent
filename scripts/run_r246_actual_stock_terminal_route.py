"""Prove one R246 terminal win through a sealed staged parent, or fail closed.

This package-external worker is intentionally narrow.  It accepts exactly one
content-addressed archived physical-game root: episode 89740321, seat 1,
step 223.  At that root the physical game's recorded legal ``END`` action
ended the archived episode with a root reward of +1 and opponent reward of
-1 while the opponent deck was empty.  The worker does not reconstruct,
alter, or annotate the state.  It supplies that archived raw observation to
the sealed ``main.agent`` exactly once and accepts a result only when the
staged parent itself emits the literal R246 deterministic terminal-win marker.

The worker is meant to run only as an auxiliary scenario below
``run_r235_r246_exact_stage_probe.py``.  That owner starts this process in a
fresh exact-child watchdog.  A native/model call which does not return is
therefore contained by the caller; this worker never starts a shell, network
client, Kaggle action, upload, queue, service, or BO workload.

The archived replay is deliberately an external, required, SHA-pinned input.
It is not copied into the sealed package and it is not a stage-deck
reachability claim.  A nonmatching or missing replay, a direct/high-confidence
route, a contained fallback, a nonterminal marker, a changed staged tree, or
any marker/proof mismatch exits nonzero without emitting a passed JSON row.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r228_kaggle_r244_harness_common import (  # noqa: I001
    COMPLETE_ACTION_CAP,
    HarnessContractError,
    as_action,
    collect_markers,
    load_binding_identity,
    prepare_exact_stage_import,
    require_module_from_exact_stage,
    sha256_file,
    stage_snapshot,
    validate_decision_marker,
)


SCHEMA = "poke_bot.r235_r246_exact_stage_scenario_evidence/v1"
WITNESS_ORIGIN = "actual_stock_search_route"
R246_STOP_REASON = "proven_deterministic_terminal_win_this_turn"
R246_PROOF_KIND = "exact_deterministic_simulator_terminal_win_this_turn"

# This source is not tracked as part of a candidate package.  Requiring an
# explicit path lets an evaluator make the source available separately while
# retaining one exact, content-addressed physical state.
EXPECTED_REPLAY_SHA256 = "sha256:270cc6481b29601fd94aa027dab1c203dda188247d3160e269c775f6a37c6a07"
EXPECTED_STEP_INDEX = 223
EXPECTED_SEAT = 1
EXPECTED_ACTION = [6]
EXPECTED_LEGAL_ORDER = [[0], [1], [2], [3], [4], [5], [6]]
MAX_PARENT_STDOUT_BYTES = 4 * 1024 * 1024


class ActualStockTerminalRouteError(RuntimeError):
    """The one reviewed physical terminal route cannot truthfully pass."""


@dataclass(frozen=True)
class ArchivedPhysicalRoot:
    """The immutable replay root and its independently recorded successor."""

    observation: dict[str, Any]
    source: dict[str, Any]


class _BoundedTee:
    """Forward staged output to stderr while bounding marker-capture memory."""

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
            # Preserve only complete UTF-8 text, which is sufficient for the
            # line-oriented marker parser.  Any truncation is a hard failure.
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


def _regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ActualStockTerminalRouteError(f"{label} must be a regular non-symlink file")
    return raw.resolve()


def _physical_directory(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ActualStockTerminalRouteError(f"{label} must be an existing physical directory")
    return raw.resolve()


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ActualStockTerminalRouteError(f"{label} is not JSON-native") from exc


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActualStockTerminalRouteError(f"{label} must be an object")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActualStockTerminalRouteError(f"{label} must be an integer")
    return int(value)


def _read_archived_physical_root(replay: Path) -> ArchivedPhysicalRoot:
    """Read and strictly bind the single reviewed real physical game root."""

    replay = _regular_file(replay, label="archived physical replay")
    observed_digest = sha256_file(replay)
    if observed_digest != EXPECTED_REPLAY_SHA256:
        raise ActualStockTerminalRouteError(
            "archived physical replay SHA-256 is not the reviewed episode-89740321 bytes"
        )
    try:
        raw = json.loads(replay.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActualStockTerminalRouteError("archived physical replay is unreadable JSON") from exc
    document = _mapping(raw, label="archived physical replay")
    steps = document.get("steps")
    if not isinstance(steps, list) or len(steps) <= EXPECTED_STEP_INDEX + 1:
        raise ActualStockTerminalRouteError("archived replay lacks the reviewed physical root/successor")
    root_row = steps[EXPECTED_STEP_INDEX]
    successor_row = steps[EXPECTED_STEP_INDEX + 1]
    if not isinstance(root_row, list) or not isinstance(successor_row, list):
        raise ActualStockTerminalRouteError("archived replay step rows are malformed")
    if len(root_row) <= EXPECTED_SEAT or len(successor_row) <= EXPECTED_SEAT:
        raise ActualStockTerminalRouteError("archived replay lacks the reviewed seat row")
    root_entry = _mapping(root_row[EXPECTED_SEAT], label="archived physical root entry")
    successor = _mapping(successor_row[EXPECTED_SEAT], label="archived physical successor")
    opponent_successor = _mapping(
        successor_row[1 - EXPECTED_SEAT], label="archived physical opponent successor"
    )
    observation_raw = _mapping(root_entry.get("observation"), label="archived physical root")
    observation = dict(_json_copy(dict(observation_raw), label="archived physical root"))
    action = as_action(root_entry.get("action"), field="archived physical root action")
    if action != EXPECTED_ACTION:
        raise ActualStockTerminalRouteError("archived root action is not the reviewed terminal END action")

    current = _mapping(observation.get("current"), label="archived root current state")
    if _integer(current.get("result"), label="archived root result") != -1:
        raise ActualStockTerminalRouteError("archived root is already terminal")
    if _integer(current.get("yourIndex"), label="archived root actor seat") != EXPECTED_SEAT:
        raise ActualStockTerminalRouteError("archived root actor is not the reviewed seat")
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise ActualStockTerminalRouteError("archived root has no two-player physical state")
    opponent = _mapping(players[1 - EXPECTED_SEAT], label="archived root opponent")
    if _integer(opponent.get("deckCount"), label="archived root opponent deck count") != 0:
        raise ActualStockTerminalRouteError("archived root opponent is not decked out")
    if opponent.get("hand") is not None:
        raise ActualStockTerminalRouteError(
            "archived root leaks opponent hand and is not a legal information-set prompt"
        )
    actor = _mapping(players[EXPECTED_SEAT], label="archived root actor")
    if _integer(actor.get("deckCount"), label="archived root actor deck count") <= 0:
        raise ActualStockTerminalRouteError("archived root actor has no physical deck remaining")

    search_begin_input = observation.get("search_begin_input")
    if not isinstance(search_begin_input, str) or not search_begin_input:
        raise ActualStockTerminalRouteError("archived root lacks its physical SearchBegin input")
    select = _mapping(observation.get("select"), label="archived root select")
    if (
        _integer(select.get("type"), label="archived root select type") != 0
        or _integer(select.get("context"), label="archived root select context") != 0
        or _integer(select.get("minCount"), label="archived root select min count") != 1
        or _integer(select.get("maxCount"), label="archived root select max count") != 1
    ):
        raise ActualStockTerminalRouteError("archived root is not the reviewed one-choice selection shape")
    options = select.get("option")
    if not isinstance(options, list) or len(options) != len(EXPECTED_LEGAL_ORDER):
        raise ActualStockTerminalRouteError("archived root option count drifted")
    option_types = [
        _integer(_mapping(option, label="archived root option").get("type"), label="option type")
        for option in options
    ]
    if option_types != [7, 7, 7, 7, 7, 7, 14]:
        raise ActualStockTerminalRouteError("archived root option order is not six plays plus END")

    successor_action = as_action(successor.get("action"), field="archived successor action")
    if successor_action != EXPECTED_ACTION:
        raise ActualStockTerminalRouteError("archived successor action does not preserve terminal END")
    if successor.get("status") != "DONE" or _integer(
        successor.get("reward"), label="archived successor reward"
    ) != 1:
        raise ActualStockTerminalRouteError("archived END successor is not a recorded win")
    if opponent_successor.get("status") != "DONE" or _integer(
        opponent_successor.get("reward"), label="archived opponent successor reward"
    ) != -1:
        raise ActualStockTerminalRouteError("archived END opponent successor is not a recorded loss")

    return ArchivedPhysicalRoot(
        observation=observation,
        source={
            "kind": "archived_stock_physical_game_replay",
            "replay_path": str(replay),
            "replay_sha256": observed_digest,
            "root_json_pointer": f"/steps/{EXPECTED_STEP_INDEX}/{EXPECTED_SEAT}/observation",
            "recorded_root_action": list(EXPECTED_ACTION),
            "recorded_successor_json_pointer": (
                f"/steps/{EXPECTED_STEP_INDEX + 1}/{EXPECTED_SEAT}"
            ),
            "root_actor_seat": EXPECTED_SEAT,
            "opponent_deck_count_at_root": 0,
            "root_legal_order_expected": [list(action) for action in EXPECTED_LEGAL_ORDER],
            "recorded_successor_status": "DONE",
            "recorded_episode_reward_winner_seat": EXPECTED_SEAT,
            "recorded_episode_reward_loser_seat": 1 - EXPECTED_SEAT,
            # Do not misrepresent a Marnie archived state as a stage-deck
            # reachability certificate.  The qualifying proof remains the
            # staged parent/stock SearchStep marker emitted below.
            "stage_deck_reachability_claimed": False,
        },
    )


def _load_exact_stage(stage: Path) -> tuple[Any, Any]:
    """Import only the supplied immutable package and disable bytecode writes."""

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    stage = prepare_exact_stage_import(stage)
    os.environ["CG_LIB_PATH"] = str(stage)
    os.chdir(stage)
    main = importlib.import_module("main")
    features = importlib.import_module("poke_bot.features")
    require_module_from_exact_stage(main, module_name="main", stage=stage)
    require_module_from_exact_stage(features, module_name="poke_bot.features", stage=stage)
    return main, features


def _complete_legal_actions(features: Any, observation: Mapping[str, Any]) -> list[list[int]]:
    try:
        raw = features.enumerate_action_combos(
            dict(_json_copy(dict(observation), label="root for legal enumeration")),
            max_combos=COMPLETE_ACTION_CAP,
        )
    except Exception as exc:
        raise ActualStockTerminalRouteError(
            "sealed feature legal-action enumeration failed under complete cap "
            f"{COMPLETE_ACTION_CAP}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        legal = [as_action(action, field="sealed complete legal action") for action in raw]
    except HarnessContractError as exc:
        raise ActualStockTerminalRouteError("sealed legal action order is malformed") from exc
    if legal != EXPECTED_LEGAL_ORDER:
        raise ActualStockTerminalRouteError(
            "sealed complete legal order does not exactly match the archived END root"
        )
    return legal


def _close_staged_broker(main: Any) -> dict[str, Any]:
    """Close only the exact broker object constructed by this staged parent."""

    broker = getattr(main, "_BROKER", None)
    if broker is None:
        return {"broker_constructed": False, "close_called": False}
    close = getattr(broker, "close", None)
    if not callable(close):
        raise ActualStockTerminalRouteError("sealed parent broker lacks a close method")
    close()
    return {"broker_constructed": True, "close_called": True}


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActualStockTerminalRouteError(f"{label} must be a finite nonnegative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ActualStockTerminalRouteError(f"{label} must be a finite nonnegative number")
    return parsed


def _require_outer_r246_terminal_fields(marker: Mapping[str, Any]) -> None:
    """Fail here if the literal marker cannot satisfy the outer exact gate.

    This is intentionally a presence/semantic check on *staged* telemetry,
    not a compatibility shim.  The worker never writes aliases into the
    marker, so an older stage continues to fail rather than being relabelled
    as R246-complete by package-external code.
    """

    required_true = (
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
    )
    missing_or_false = [field for field in required_true if marker.get(field) is not True]
    if missing_or_false:
        raise ActualStockTerminalRouteError(
            "literal staged r246 marker lacks required exact-gate true fields: "
            + ", ".join(missing_or_false)
        )
    for field in (
        "completed_root_backup_count",
        "terminal_win_proof_count",
        "proven_deterministic_terminal_win_this_turn_stop_count",
    ):
        if _integer(marker.get(field), label=f"literal staged r246 {field}") < 1:
            raise ActualStockTerminalRouteError(
                f"literal staged r246 marker {field} is not positive"
            )
    _finite_nonnegative(
        marker.get("child_search_elapsed_seconds"),
        label="literal staged r246 child search elapsed",
    )
    _finite_nonnegative(
        marker.get("parent_action_elapsed_seconds"),
        label="literal staged r246 parent action elapsed",
    )


def _require_literal_r246_marker(
    *,
    staged_stdout: str,
    action: Sequence[int],
    legal_actions: Sequence[Sequence[int]],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept the exact staged r246 receipt, never a worker-built substitute."""

    try:
        markers = collect_markers(staged_stdout)
    except HarnessContractError as exc:
        raise ActualStockTerminalRouteError("sealed parent emitted malformed marker JSON") from exc
    if markers["hard_failures"]:
        raise ActualStockTerminalRouteError("sealed parent emitted a hard-failure marker")
    if markers["degraded_fallbacks"]:
        raise ActualStockTerminalRouteError("sealed parent emitted a contained fallback marker")
    if markers["full_gameplay_successes"]:
        raise ActualStockTerminalRouteError("single nonterminal root unexpectedly emitted a full-game marker")
    decisions = markers["decisions"]
    if len(decisions) != 1:
        raise ActualStockTerminalRouteError(
            "sealed main.agent did not emit exactly one decision marker for the physical root"
        )
    marker = dict(decisions[0])
    try:
        validated = validate_decision_marker(
            marker,
            legal_actions=legal_actions,
            observation=observation,
        )
    except HarnessContractError as exc:
        raise ActualStockTerminalRouteError(
            f"sealed decision marker does not satisfy r242/r244/r246 validation: {exc}"
        ) from exc
    if marker.get("mode") != "shared_tree_mcts":
        raise ActualStockTerminalRouteError("sealed parent did not take the ambiguous MCTS route")
    if marker.get("stop_reason") != R246_STOP_REASON:
        raise ActualStockTerminalRouteError("sealed MCTS route did not stop for a deterministic terminal win")
    selected = as_action(marker.get("selected_action"), field="sealed marker selected action")
    returned = as_action(action, field="sealed parent returned action")
    if selected != returned or selected != EXPECTED_ACTION:
        raise ActualStockTerminalRouteError(
            "sealed parent did not select the reviewed END action from the actual r246 marker"
        )
    if marker.get("terminal_leaf_returned_by_exact_stock_simulator") is not True:
        raise ActualStockTerminalRouteError(
            "sealed r246 marker does not prove terminal leaf return by exact stock simulator"
        )
    if marker.get("mcts_action_authority") is not True:
        raise ActualStockTerminalRouteError("sealed r246 marker lacks MCTS action authority")
    if marker.get("degraded") is not False:
        raise ActualStockTerminalRouteError("sealed r246 marker is degraded")
    _require_outer_r246_terminal_fields(marker)
    proof = validated.get("terminal_win_proof")
    if not isinstance(proof, Mapping):
        raise ActualStockTerminalRouteError("sealed r246 marker lacks a validated terminal proof")
    if proof.get("proof_kind") != R246_PROOF_KIND:
        raise ActualStockTerminalRouteError("sealed r246 proof kind drifted")
    if proof.get("root_action") != EXPECTED_ACTION or proof.get("selected_action") != EXPECTED_ACTION:
        raise ActualStockTerminalRouteError("sealed r246 proof action drifted from archived END")
    return marker


def _run(
    *,
    stage: Path,
    candidate_archive: Path,
    member_manifest: Path,
    r225_contract: Path,
    r236_contract: Path,
    source_replay: Path,
) -> dict[str, Any]:
    """Run the sole physical root through the sealed parent once, or raise."""

    stage = _physical_directory(stage, label="sealed stage")
    candidate_archive = _regular_file(candidate_archive, label="candidate archive")
    member_manifest = _regular_file(member_manifest, label="member manifest")
    r225_contract = _regular_file(r225_contract, label="r225 contract")
    r236_contract = _regular_file(r236_contract, label="r236 contract")
    archived_root = _read_archived_physical_root(source_replay)
    try:
        binding = load_binding_identity(
            stage=stage,
            candidate_archive=candidate_archive,
            member_manifest=member_manifest,
            r225_contract=r225_contract,
            r236_contract=r236_contract,
        )
    except Exception as exc:
        raise ActualStockTerminalRouteError(f"exact archive/stage binding failed: {exc}") from exc
    before = stage_snapshot(stage)

    tee = _BoundedTee(sys.stderr, max_bytes=MAX_PARENT_STDOUT_BYTES)
    staged_main: Any | None = None
    body_error: Exception | None = None
    close_error: Exception | None = None
    action: list[int] | None = None
    legal_actions: list[list[int]] | None = None
    marker: dict[str, Any] | None = None
    elapsed_seconds: float | None = None
    broker_close: dict[str, Any] | None = None
    try:
        with contextlib.redirect_stdout(tee):
            staged_main, features = _load_exact_stage(stage)
            legal_actions = _complete_legal_actions(features, archived_root.observation)
            started = time.monotonic()
            action = as_action(
                staged_main.agent(
                    dict(_json_copy(archived_root.observation, label="sealed parent root"))
                ),
                field="sealed parent action",
            )
            elapsed_seconds = max(0.0, time.monotonic() - started)
            if tee.truncated:
                raise ActualStockTerminalRouteError(
                    "sealed parent stdout exceeded the bounded marker-capture limit"
                )
            marker = _require_literal_r246_marker(
                staged_stdout=tee.text,
                action=action,
                legal_actions=legal_actions,
                observation=archived_root.observation,
            )
    except Exception as exc:  # noqa: BLE001 - preserve exact broker close/snapshot
        body_error = exc
    finally:
        with contextlib.redirect_stdout(tee):
            if staged_main is not None:
                try:
                    broker_close = _close_staged_broker(staged_main)
                except Exception as exc:  # noqa: BLE001 - owned child cleanup is fatal
                    close_error = exc

    try:
        after = stage_snapshot(stage)
    except BaseException as exc:
        raise ActualStockTerminalRouteError("sealed stage cannot be snapshotted after scenario") from exc
    if body_error is not None:
        raise body_error
    if close_error is not None:
        raise ActualStockTerminalRouteError("sealed staged broker did not close cleanly") from close_error
    if tee.truncated:
        raise ActualStockTerminalRouteError("sealed parent stdout exceeded the bounded marker-capture limit")
    if after != before:
        raise ActualStockTerminalRouteError("actual-stock route mutated the sealed stage")
    if (
        action is None
        or legal_actions is None
        or marker is None
        or elapsed_seconds is None
        or broker_close is None
    ):
        raise ActualStockTerminalRouteError("actual-stock route did not complete its required evidence")

    # Keep the exact parent marker verbatim.  The outer R235 converter may
    # normalize only spelling aliases; it must never derive a terminal proof
    # from this worker's source facts.
    return {
        "schema": SCHEMA,
        "status": "passed",
        "passed": True,
        "witness_origin": WITNESS_ORIGIN,
        "common_identity": dict(binding["common_identity"]),
        "exact_package_identity": dict(binding["exact_package"]),
        "stage_contract": dict(binding["stage_contract"]),
        "r240_witnesses": {
            "synthetic_proven_deterministic_terminal_win_this_turn": marker,
        },
        "physical_root_source": archived_root.source,
        "sealed_parent": {
            "entrypoint": str(stage / "main.py"),
            "agent_call_count": 1,
            "returned_action": action,
            "complete_legal_order": legal_actions,
            "elapsed_seconds": elapsed_seconds,
            "broker_close": broker_close,
            "parent_stdout_capture_truncated": False,
        },
        "stage_mutation_check": {
            "before_tree_sha256": before["tree_sha256"],
            "after_tree_sha256": after["tree_sha256"],
            "unchanged": True,
        },
        "network_accessed": False,
        "kaggle_api_called": False,
        "kaggle_upload_used": False,
        "kaggle_queue_used": False,
        "bo_workload_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument("--r225-contract", type=Path, required=True)
    parser.add_argument("--r236-contract", type=Path, required=True)
    parser.add_argument(
        "--source-replay",
        type=Path,
        required=True,
        help=(
            "physical episode-89740321 replay; its exact SHA-256 is pinned by this worker"
        ),
    )
    args = parser.parse_args()
    try:
        # ``stdout`` is a one-object protocol for the outer scenario runner.
        # Keep even an unexpected helper/import print on stderr so it cannot
        # turn a successful typed witness into ambiguous stdout framing.
        with contextlib.redirect_stdout(sys.stderr):
            payload = _run(
                stage=args.stage,
                candidate_archive=args.candidate_archive,
                member_manifest=args.member_manifest,
                r225_contract=args.r225_contract,
                r236_contract=args.r236_contract,
                source_replay=args.source_replay,
            )
    except Exception as exc:  # noqa: BLE001 - stdout is a one-object protocol
        print(
            "R246_ACTUAL_STOCK_TERMINAL_ROUTE_FAILED_CLOSED "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
