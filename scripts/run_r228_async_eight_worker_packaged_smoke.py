#!/usr/bin/env python3
"""Run one immutable r244 package through one complete physical stock game.

This is a package-external evaluator.  It imports ``main.py`` only from the
provided sealed stage, drives one real ``battle_start(deck, deck)`` game to a
stock terminal state, and verifies the r238/r242/r244 receipt contract.  It
never creates a Kaggle submission, queue item, service, GPU job, or package
mutation.  A caller must still put this process under an exact-child watchdog:
Python cannot interrupt a native call that never returns.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import platform
import resource
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from r228_kaggle_r244_harness_common import (
    COMPLETE_ACTION_CAP,
    DECISION_PREFIX,
    DEGRADED_FALLBACK_PREFIX,
    FULL_GAME_RECEIPT_NAME,
    FULL_GAMEPLAY_SUCCESS_PREFIX,
    HARD_FAILURE_PREFIX,
    HarnessContractError,
    as_action,
    collect_markers,
    load_binding_identity,
    passed_preflight_receipt,
    prepare_exact_stage_import,
    require,
    require_module_from_exact_stage,
    stage_snapshot,
    validate_decision_marker,
    validate_degraded_marker,
    validate_full_game_success,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "poke_bot.r244_exact_package_full_game_smoke/v1"
FINAL_PREFIX = "R244_EXACT_PACKAGE_FULL_GAME_SMOKE "
RAW_R240_PROBE_SCHEMA = "poke_bot.r244_exact_package_raw_physical_game_telemetry/v1"


class SmokeError(RuntimeError):
    """The sealed full-game smoke did not satisfy a fail-closed invariant."""


class _Tee:
    def __init__(self, *targets: TextIO) -> None:
        self._targets = targets
        self._parts: list[str] = []

    def write(self, value: str) -> int:
        self._parts.append(value)
        for target in self._targets:
            target.write(value)
        return len(value)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()

    @property
    def text(self) -> str:
        return "".join(self._parts)


@dataclass
class _Journal:
    started_monotonic: float = field(default_factory=time.monotonic)
    stage_contract: dict[str, Any] = field(default_factory=dict)
    binding_identity: dict[str, Any] = field(default_factory=dict)
    package_before: dict[str, Any] = field(default_factory=dict)
    package_after: dict[str, Any] = field(default_factory=dict)
    battle_started: bool = False
    battle_finish_called: bool = False
    broker_close_called: bool = False
    action_calls: list[dict[str, Any]] = field(default_factory=list)
    terminal_boundary_callback: dict[str, Any] = field(default_factory=dict)
    markers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    process_observation_start: dict[str, Any] = field(default_factory=dict)
    process_observation_peak: dict[str, Any] = field(default_factory=dict)
    process_observation_end: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""


def _thread_observation() -> dict[str, Any]:
    """Read process thread count without changing any runtime setting."""

    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        try:
            for line in proc_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("Threads:"):
                    return {
                        "count": int(line.split(":", 1)[1].strip()),
                        "source": "proc_self_status",
                    }
        except (OSError, ValueError):
            pass
    return {"count": threading.active_count(), "source": "python_threading_active_count"}


def _process_observation() -> dict[str, Any]:
    """Capture raw host measurements, never an inferred submission envelope."""

    try:
        raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_error = None
    except (OSError, ValueError) as exc:
        raw_rss = None
        rss_error = f"{type(exc).__name__}: {exc}"
    return {
        "platform": platform.system().lower(),
        "ru_maxrss_raw": raw_rss,
        # POSIX exposes KiB on Linux and bytes on macOS.  Preserve the raw
        # source rather than silently normalizing an uncertain unit.
        "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "kib",
        "ru_maxrss_error": rss_error,
        "thread": _thread_observation(),
    }


def _stage_disk_bytes(stage: Path) -> int:
    total = 0
    for path in stage.rglob("*"):
        if path.is_symlink():
            raise SmokeError("sealed stage acquired a symlink during raw telemetry")
        if path.is_file():
            total += path.stat().st_size
    return total


def _marker_cuda_observations(markers: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    parent: list[dict[str, Any]] = []
    child: list[dict[str, Any]] = []
    for group in ("decisions", "degraded_fallbacks", "full_gameplay_successes"):
        for marker in markers.get(group, []):
            parent_observation = marker.get("parent_cuda_runtime_before_search")
            if isinstance(parent_observation, dict):
                parent.append(dict(parent_observation))
            broker = marker.get("broker")
            if isinstance(broker, dict):
                identity = broker.get("child_identity")
                if isinstance(identity, dict):
                    child_observation = identity.get("cuda_runtime_before_search")
                    if isinstance(child_observation, dict):
                        child.append(dict(child_observation))
            identity = marker.get("child_identity")
            if isinstance(identity, dict):
                child_observation = identity.get("cuda_runtime_before_search")
                if isinstance(child_observation, dict):
                    child.append(dict(child_observation))
    return {"parent": parent, "child": child}


def _raw_r240_probe_envelope(
    *, payload: dict[str, Any], journal: _Journal, receipt: Path
) -> dict[str, Any]:
    """Expose actual physical-game evidence for the separate R240 converter.

    This is deliberately not the final R240 probe schema: the smoke game
    cannot promise that one naturally occurring game exercises every required
    high-direct, continuation, and r246 proof route.  The consumer therefore
    owns route coverage/classification and must fail closed when a witness is
    absent.
    """

    markers = payload.get("markers")
    marker_map = markers if isinstance(markers, dict) else {}
    callbacks = payload.get("stock_game", {}).get("actions", [])
    raw_stage = payload.get("stage")
    stage_disk_bytes: int | None = None
    if isinstance(raw_stage, str):
        stage_path = Path(raw_stage)
        if stage_path.is_dir() and not stage_path.is_symlink():
            stage_disk_bytes = _stage_disk_bytes(stage_path)
    return {
        "schema": RAW_R240_PROBE_SCHEMA,
        "status": payload.get("status"),
        "harness_schema": SCHEMA,
        "receipt_path": str(receipt),
        "exact_package_identity": payload.get("exact_package_identity"),
        "stage_contract": payload.get("stage_contract"),
        "package_mutation_check": payload.get("package_mutation_check"),
        "stage_disk_bytes": stage_disk_bytes,
        "stock_game": payload.get("stock_game"),
        "callbacks": callbacks,
        "decision_markers": marker_map.get("decisions", []),
        "degraded_fallback_markers": marker_map.get("degraded_fallbacks", []),
        "hard_failure_markers": marker_map.get("hard_failures", []),
        "full_game_success_markers": marker_map.get("full_gameplay_successes", []),
        "cuda_observations": _marker_cuda_observations(marker_map),
        "process_observation": {
            "start": journal.process_observation_start,
            "peak": journal.process_observation_peak,
            "end": journal.process_observation_end,
        },
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "failure": payload.get("failure"),
    }


def _deck(stage: Path) -> list[int]:
    cards: list[int] = []
    for raw in (stage / "deck.csv").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            cards.append(int(line.split(",", 1)[0]))
        if len(cards) == 60:
            break
    require(len(cards) == 60, "packaged deck is not 60 cards")
    return cards


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _write_receipt_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeError(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SmokeError(f"temporary receipt path already exists: {temporary}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # ``link`` publishes this inode unchanged, so set the immutable mode
        # before the full-game receipt becomes observable at its final path.
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SmokeError(f"receipt already exists: {path}") from exc
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _complete_legal_actions(features: Any, observation: dict[str, Any]) -> list[list[int]] | None:
    if observation.get("select") is None:
        return None
    try:
        raw = features.enumerate_action_combos(
            dict(observation), max_combos=COMPLETE_ACTION_CAP
        )
    except Exception as exc:
        raise SmokeError(
            "complete legal action enumeration failed under "
            f"complete_action_cap={COMPLETE_ACTION_CAP}: {type(exc).__name__}: {exc}"
        ) from exc
    legal = [as_action(row, field="complete legal action") for row in raw]
    require(bool(legal), "active stock prompt has no complete legal actions")
    return legal


def _validate_action(
    *, action: list[int], legal: list[list[int]] | None, deck: list[int]
) -> None:
    if legal is None:
        require(action == deck, "deck/terminal callback did not return the packaged deck")
    else:
        require(action in legal, "packaged action is outside the complete legal order")


def _check_elapsed(*, started: float, timeout_seconds: float, phase: str) -> float:
    elapsed = time.monotonic() - started
    if elapsed > timeout_seconds:
        raise SmokeError(f"{phase} exceeded its {timeout_seconds:.3f}s hard bound")
    return elapsed


def _load_stage(stage: Path) -> tuple[Any, Any, Any]:
    """Import only the provided stage and disable Python bytecode writes."""

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    stage = prepare_exact_stage_import(stage)
    stage_text = str(stage)
    os.environ["CG_LIB_PATH"] = stage_text
    os.chdir(stage_text)
    main = importlib.import_module("main")
    require_module_from_exact_stage(main, module_name="main", stage=stage)
    poke_bot = importlib.import_module("poke_bot")
    require_module_from_exact_stage(poke_bot, module_name="poke_bot", stage=stage)
    from poke_bot import cg_env, features

    require_module_from_exact_stage(cg_env, module_name="poke_bot.cg_env", stage=stage)
    require_module_from_exact_stage(features, module_name="poke_bot.features", stage=stage)
    return main, cg_env, features


def _new_marker_slice(tee: _Tee, before: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    all_markers = collect_markers(tee.text)
    result: dict[str, list[dict[str, Any]]] = {}
    for kind, rows in all_markers.items():
        prior = before.get(kind, 0)
        require(len(rows) >= prior, f"{kind} marker count moved backwards")
        result[kind] = rows[prior:]
    return result


def _marker_counts(tee: _Tee) -> dict[str, int]:
    return {kind: len(rows) for kind, rows in collect_markers(tee.text).items()}


def _validate_callback_markers(
    *,
    legal: list[list[int]] | None,
    observation: dict[str, Any] | None,
    callback_markers: dict[str, list[dict[str, Any]]],
    previous_degraded_count: int,
) -> list[dict[str, Any]]:
    """Bind zero/one decision callback to r242/r244 receipt authority."""

    require(
        not callback_markers["hard_failures"],
        "package emitted an r238 hard-failure marker",
    )
    require(
        len(callback_markers["decisions"]) <= 1,
        "one callback emitted multiple decision receipts",
    )
    require(
        len(callback_markers["degraded_fallbacks"]) <= 1,
        "one callback emitted multiple containment receipts",
    )
    require(
        not callback_markers["full_gameplay_successes"],
        "full-game success marker appeared before terminal boundary",
    )
    validated: list[dict[str, Any]] = []
    if legal is None:
        require(
            not callback_markers["decisions"] and not callback_markers["degraded_fallbacks"],
            "deck callback emitted a decision/containment receipt",
        )
        return validated
    for marker in callback_markers["decisions"]:
        validated.append(
            validate_decision_marker(
                marker, legal_actions=legal, observation=observation
            )
        )
    for marker in callback_markers["degraded_fallbacks"]:
        validated.append(
            {
                "mode": "contained_child_fault",
                "degraded": True,
                **validate_degraded_marker(marker, legal_actions=legal),
            }
        )
    # A game latched direct-only after a previously validated contained fault
    # intentionally produces no new native receipt on later decisions.
    if len(legal) > 1 and not validated:
        require(
            previous_degraded_count > 0,
            "branching callback emitted neither r242/r244 receipt nor containment marker",
        )
    return validated


def _close_exact_package_broker(main: Any) -> bool:
    broker = getattr(main, "_BROKER", None)
    if broker is None:
        return False
    close = getattr(broker, "close", None)
    require(callable(close), "staged broker lacks close")
    close()
    return True


def _run(
    stage: Path,
    *,
    candidate_archive: Path,
    member_manifest: Path,
    r225_contract: Path,
    r236_contract: Path,
    max_actions: int,
    game_timeout_seconds: float,
    per_action_timeout_seconds: float,
    marker_stream: TextIO,
    journal: _Journal,
) -> dict[str, Any]:
    stage = stage.resolve()
    try:
        journal.binding_identity = load_binding_identity(
            stage=stage,
            candidate_archive=candidate_archive,
            member_manifest=member_manifest,
            r225_contract=r225_contract,
            r236_contract=r236_contract,
        )
        journal.stage_contract = dict(journal.binding_identity["stage_contract"])
        journal.package_before = dict(journal.stage_contract["stage_snapshot"])
        journal.process_observation_start = _process_observation()
        main, cg_env, features = _load_stage(stage)
        deck = _deck(stage)
        tee = _Tee(marker_stream)
        game_started = time.monotonic()
        observation, started = cg_env.battle_start(deck, deck)
        if observation is None:
            raise SmokeError(f"BattleStart failed: {getattr(started, 'errorType', None)}")
        journal.battle_started = True
        _check_elapsed(
            started=game_started,
            timeout_seconds=game_timeout_seconds,
            phase="BattleStart",
        )

        while not cg_env.is_finished(observation):
            require(
                len(journal.action_calls) < max_actions,
                f"stock game exceeded action ceiling {max_actions}",
            )
            _check_elapsed(
                started=game_started,
                timeout_seconds=game_timeout_seconds,
                phase="before callback",
            )
            legal = _complete_legal_actions(features, observation)
            counts_before = _marker_counts(tee)
            call_started = time.monotonic()
            with contextlib.redirect_stdout(tee):
                action = as_action(main.agent(dict(observation)), field="packaged action")
            callback_elapsed = _check_elapsed(
                started=call_started,
                timeout_seconds=per_action_timeout_seconds,
                phase="agent callback",
            )
            _check_elapsed(
                started=game_started,
                timeout_seconds=game_timeout_seconds,
                phase="after callback",
            )
            _validate_action(action=action, legal=legal, deck=deck)
            callback_markers = _new_marker_slice(tee, counts_before)
            validated = _validate_callback_markers(
                legal=legal,
                observation=observation,
                callback_markers=callback_markers,
                previous_degraded_count=counts_before["degraded_fallbacks"],
            )
            authority_markers = (
                callback_markers["decisions"]
                + callback_markers["degraded_fallbacks"]
            )
            require(
                len(authority_markers) <= 1,
                "callback emitted more than one raw action-authority marker",
            )
            journal.action_calls.append(
                {
                    "call_index": len(journal.action_calls),
                    "observation_step": observation.get("step"),
                    "legal_action_count": None if legal is None else len(legal),
                    "action": action,
                    "callback_elapsed_seconds": callback_elapsed,
                    "receipt_modes": [item["mode"] for item in validated],
                    # Preserve the actual package marker in callback order for
                    # the external R240 probe converter.  It is null only for
                    # an allowed post-containment direct-only callback.
                    "decision_marker_or_containment": (
                        authority_markers[0] if authority_markers else None
                    ),
                    "raw_decision_markers": callback_markers["decisions"],
                    "raw_containment_markers": callback_markers["degraded_fallbacks"],
                    "stock_action_accepted": False,
                }
            )
            observation = cg_env.battle_select(action)
            journal.action_calls[-1]["stock_action_accepted"] = True
            _check_elapsed(
                started=game_started,
                timeout_seconds=game_timeout_seconds,
                phase="after BattleSelect",
            )

        # Only after stock libcg declares the physical game terminal may this
        # evaluator invoke the package's next deck-boundary callback.  That
        # callback is how the staged entrypoint emits its one full-game marker;
        # no action is sent back into the already finished physical game.
        require(cg_env.is_finished(observation), "stock game did not reach terminal state")
        counts_before_terminal = _marker_counts(tee)
        terminal = dict(observation)
        terminal["select"] = None
        terminal_started = time.monotonic()
        with contextlib.redirect_stdout(tee):
            terminal_action = as_action(main.agent(terminal), field="terminal deck action")
        terminal_elapsed = _check_elapsed(
            started=terminal_started,
            timeout_seconds=per_action_timeout_seconds,
            phase="terminal deck-boundary callback",
        )
        require(terminal_action == deck, "terminal deck-boundary callback did not return package deck")
        terminal_markers = _new_marker_slice(tee, counts_before_terminal)
        require(not terminal_markers["decisions"], "terminal boundary emitted a new decision receipt")
        require(not terminal_markers["degraded_fallbacks"], "terminal boundary emitted a new containment marker")
        require(not terminal_markers["hard_failures"], "terminal boundary emitted hard failure")
        require(
            len(terminal_markers["full_gameplay_successes"]) == 1,
            "true stock terminal boundary did not emit exactly one success marker",
        )
        journal.terminal_boundary_callback = {
            "stock_terminal_before_callback": True,
            "select": None,
            "returned_action": terminal_action,
            "callback_elapsed_seconds": terminal_elapsed,
            "new_full_gameplay_success_markers": 1,
        }

        journal.markers = collect_markers(tee.text)
        journal.stdout = tee.text
        journal.process_observation_peak = _process_observation()
        require(not journal.markers["hard_failures"], "package emitted hard-failure marker")
        require(
            not journal.markers["degraded_fallbacks"],
            "full-game smoke observed a contained degraded fallback; it has no viability-success credit",
        )
        require(
            len(journal.markers["full_gameplay_successes"]) == 1,
            "package emitted other than exactly one terminal success marker",
        )
        mcts_count = sum(
            1
            for marker in journal.markers["decisions"]
            if marker.get("mode") == "shared_tree_mcts"
        )
        require(mcts_count >= 1, "full game never exercised an ambiguous two-lane MCTS decision")
        validate_full_game_success(
            journal.markers["full_gameplay_successes"][0],
            mcts_decision_count=mcts_count,
        )
        journal.broker_close_called = _close_exact_package_broker(main)
        journal.package_after = stage_snapshot(stage)
        require(
            journal.package_after == journal.package_before,
            "full-game evaluator mutated the sealed package tree",
        )
        return {
            **passed_preflight_receipt(
                receipt_name=FULL_GAME_RECEIPT_NAME,
                common_identity=journal.binding_identity["common_identity"],
                harness_schema=SCHEMA,
            ),
            "scope": "exact_r236_r238_r242_r244_package_full_physical_game",
            "stage": str(stage),
            "exact_package_identity": journal.binding_identity["exact_package"],
            "elapsed_seconds": max(0.0, time.monotonic() - game_started),
            "game_timeout_seconds": game_timeout_seconds,
            "per_action_timeout_seconds": per_action_timeout_seconds,
            "max_actions": max_actions,
            "stage_contract": journal.stage_contract,
            "package_mutation_check": {
                "before": journal.package_before,
                "after": journal.package_after,
                "unchanged": True,
            },
            "stock_game": {
                "deck": deck,
                "actions": journal.action_calls,
                "terminal": {
                    "winner": cg_env.result_winner(observation),
                    "steps": observation.get("step"),
                    "physical_terminal_confirmed": True,
                },
            },
            "markers": journal.markers,
            "terminal_boundary_callback": journal.terminal_boundary_callback,
            "broker_close_called": journal.broker_close_called,
            "exact_package_full_local_game_passed": True,
            "full_gameplay_loop_completed": True,
            "branching_gameplay_decision_count": len(journal.markers["decisions"]),
            "explicit_success_marker_count": 1,
            "degraded_game_count": 0,
            "active_simulator_search_lane_count": 2,
            "process_observation": {
                "start": journal.process_observation_start,
                "peak": journal.process_observation_peak,
            },
        }
    finally:
        try:
            if journal.battle_started:
                # This is the stock environment opened by this exact evaluator
                # process, never a managed service or an interactive session.
                cg_env.battle_finish()  # type: ignore[name-defined]
                journal.battle_finish_called = True
        finally:
            if "tee" in locals():
                journal.stdout = tee.text
            journal.process_observation_end = _process_observation()
            if not journal.package_after and stage.is_dir() and not stage.is_symlink():
                try:
                    journal.package_after = stage_snapshot(stage)
                except Exception:
                    pass


def _failure_payload(
    *,
    stage: Path,
    candidate_archive: Path,
    member_manifest: Path,
    max_actions: int,
    game_timeout_seconds: float,
    per_action_timeout_seconds: float,
    journal: _Journal,
    error: BaseException,
) -> dict[str, Any]:
    try:
        markers = collect_markers(journal.stdout)
    except Exception as marker_error:  # noqa: BLE001
        markers = {"parse_error": f"{type(marker_error).__name__}: {marker_error}"}
    unchanged = bool(journal.package_before) and journal.package_after == journal.package_before
    return {
        "schema": SCHEMA,
        "status": "failed_closed",
        "scope": "exact_r236_r238_r242_r244_package_full_physical_game",
        "stage": str(stage.resolve()),
        "candidate_archive": str(candidate_archive.resolve()),
        "member_manifest": str(member_manifest.resolve()),
        "binding_identity": journal.binding_identity,
        "elapsed_seconds": max(0.0, time.monotonic() - journal.started_monotonic),
        "game_timeout_seconds": game_timeout_seconds,
        "per_action_timeout_seconds": per_action_timeout_seconds,
        "max_actions": max_actions,
        "stage_contract": journal.stage_contract,
        "package_mutation_check": {
            "before": journal.package_before,
            "after": journal.package_after,
            "unchanged": unchanged,
        },
        "stock_game": {
            "actions": journal.action_calls,
            "battle_started": journal.battle_started,
            "battle_finish_called": journal.battle_finish_called,
        },
        "markers": markers,
        "terminal_boundary_callback": journal.terminal_boundary_callback,
        "broker_close_called": journal.broker_close_called,
        "process_observation": {
            "start": journal.process_observation_start,
            "peak": journal.process_observation_peak,
            "end": journal.process_observation_end,
        },
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument(
        "--r225-contract",
        type=Path,
        default=ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
    )
    parser.add_argument(
        "--r236-contract",
        type=Path,
        default=ROOT / "state/canonical-libcg-r236.json",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--game-timeout-seconds", type=_positive_seconds, default=900.0)
    parser.add_argument("--per-action-timeout-seconds", type=_positive_seconds, default=4.0)
    parser.add_argument(
        "--emit-r240-probe",
        action="store_true",
        help=(
            "emit one raw physical-game telemetry JSON object to stdout and send "
            "package markers/logs to stderr for the external R240 converter"
        ),
    )
    args = parser.parse_args()
    require(int(args.max_actions) >= 1, "--max-actions must be positive")
    receipt = args.receipt.resolve()
    require(not receipt.exists() and not receipt.is_symlink(), f"receipt already exists: {receipt}")

    journal = _Journal()
    marker_stream: TextIO = sys.stderr if args.emit_r240_probe else sys.stdout
    stdout_redirect = contextlib.redirect_stdout(sys.stderr) if args.emit_r240_probe else contextlib.nullcontext()
    with stdout_redirect:
        try:
            payload = _run(
                args.stage,
                candidate_archive=args.candidate_archive,
                member_manifest=args.member_manifest,
                r225_contract=args.r225_contract,
                r236_contract=args.r236_contract,
                max_actions=int(args.max_actions),
                game_timeout_seconds=float(args.game_timeout_seconds),
                per_action_timeout_seconds=float(args.per_action_timeout_seconds),
                marker_stream=marker_stream,
                journal=journal,
            )
            exit_code = 0
        except Exception as exc:  # noqa: BLE001 - seal a diagnostic failure receipt
            payload = _failure_payload(
                stage=args.stage,
                candidate_archive=args.candidate_archive,
                member_manifest=args.member_manifest,
                max_actions=int(args.max_actions),
                game_timeout_seconds=float(args.game_timeout_seconds),
                per_action_timeout_seconds=float(args.per_action_timeout_seconds),
                journal=journal,
                error=exc,
            )
            exit_code = 1
    process = payload.get("process_observation")
    if isinstance(process, dict):
        process["end"] = journal.process_observation_end
    _write_receipt_once(receipt, payload)
    if args.emit_r240_probe:
        raw_probe = _raw_r240_probe_envelope(
            payload=payload, journal=journal, receipt=receipt
        )
        print(json.dumps(raw_probe, sort_keys=True), flush=True)
    else:
        print(FINAL_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
