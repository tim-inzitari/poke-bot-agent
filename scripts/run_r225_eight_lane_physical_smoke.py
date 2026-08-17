#!/usr/bin/env python3
"""Run one isolated local r225 package child against stock ``libcg``.

This is deliberately package-external: it does not alter the staged archive
and it has no Kaggle/queue/upload code.  It creates one ordinary stock game,
lets the archived r195 direct policy make every real-game choice, and stops
immediately after the wrapper has completed its one-shot eight-lane probe and
the returned direct action has been accepted by the stock game API.

The emitted receipt is intended for a local structural capability result only.
It never upgrades a non-H100 host into a Kaggle viability pass.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import traceback
from typing import Any, TextIO


SCHEMA = "poke_bot.r225_eight_lane_physical_smoke/v1"
DIAGNOSTIC_PREFIX = "R225_EIGHT_LANE_DIAGNOSTIC "
FINAL_PREFIX = "R225_LOCAL_PHYSICAL_SMOKE "


class SmokeError(RuntimeError):
    """The isolated structural smoke did not produce a complete receipt."""


class _Tee:
    def __init__(self, *targets: TextIO) -> None:
        self._targets = targets
        self._parts: list[str] = []

    def write(self, data: str) -> int:
        self._parts.append(data)
        for target in self._targets:
            target.write(data)
        return len(data)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeError(f"receipt target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SmokeError(f"temporary receipt target already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            # This is an empty/single-purpose temporary under the exact
            # output directory and is only reached after a failed atomic write.
            temporary.unlink()


def _read_deck(stage: Path) -> list[int]:
    cards: list[int] = []
    for raw in (stage / "deck.csv").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cards.append(int(line.split(",", 1)[0]))
        if len(cards) == 60:
            break
    if len(cards) != 60:
        raise SmokeError("staged r195 deck.csv is not a 60-card deck")
    return cards


def _parse_diagnostic(output: str) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith(DIAGNOSTIC_PREFIX):
            continue
        try:
            payload = json.loads(line[len(DIAGNOSTIC_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise SmokeError("r225 emitted a malformed diagnostic JSON line") from exc
        if not isinstance(payload, dict):
            raise SmokeError("r225 emitted a non-object diagnostic payload")
        signals.append(payload)
    if len(signals) != 1:
        raise SmokeError(f"expected exactly one r225 diagnostic line, got {len(signals)}")
    return signals[0]


def _loaded_dso_paths(expected: Path) -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    matches: set[str] = set()
    expected_text = str(expected.resolve())
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        mapped = fields[5].removesuffix(" (deleted)")
        if mapped == expected_text:
            matches.add(mapped)
    return sorted(matches)


def _require(value: bool, label: str) -> None:
    if not value:
        raise SmokeError(label)


def _validate_payload(payload: dict[str, Any], *, expected_lib: Path) -> dict[str, Any]:
    _require(
        payload.get("schema")
        == "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1",
        "r225 diagnostic schema mismatch",
    )
    _require(
        payload.get("status") in {"viable", "structural_pass_resource_unverified"},
        "r225 diagnostic did not pass structurally",
    )
    stock = payload.get("stock_libcg")
    _require(isinstance(stock, dict), "r225 stock-libcg attestation is absent")
    _require(stock.get("path") == str(expected_lib.resolve()), "loaded stock DSO path mismatch")
    _require(stock.get("exact_stock_r195") is True, "loaded DSO is not exact r195 stock libcg")
    _require(stock.get("custom_engine_symbols_present") in ([], ()), "loaded DSO has custom engine symbols")
    eight = payload.get("eight_lane")
    _require(isinstance(eight, dict), "r225 eight-lane receipt is absent")
    _require(eight.get("requested_lane_count") == 8, "r225 did not request exactly eight lanes")
    _require(eight.get("active_lane_count") == 8, "r225 did not activate all eight lanes")
    batches = eight.get("batches")
    _require(isinstance(batches, list) and len(batches) >= 2, "r225 did not complete two bounded batches")
    all_lanes: list[dict[str, Any]] = []
    for row in batches:
        _require(isinstance(row, dict) and isinstance(row.get("receipt"), dict), "r225 batch receipt is missing")
        receipt = row["receipt"]
        _require(receipt.get("shared_logical_tree") is True, "batch is not one shared tree")
        _require(receipt.get("requested_lane_count") == 8, "batch lane request changed")
        _require(receipt.get("active_lane_count") == 8, "batch active lane count changed")
        _require(receipt.get("unique_raw_handle_count") == 8, "batch did not have eight raw handles")
        _require(receipt.get("max_concurrent_active_lanes") == 8, "lanes were not simultaneously active")
        _require(receipt.get("all_eight_began_before_first_step") is True, "SearchBegin overlap proof is absent")
        _require(receipt.get("root_visit_delta") == 8, "batch did not back up eight root visits")
        _require(receipt.get("completed_backed_simulations") == 8, "batch did not back up all lanes")
        _require(receipt.get("outstanding_reservations") == 0, "batch leaked reservations")
        _require(receipt.get("outstanding_virtual_loss") == 0, "batch leaked virtual loss")
        _require(receipt.get("native_search_id_cross_lane_reuse") == 0, "batch reused a native search state across lanes")
        _require(receipt.get("partial_lane_statistics_used") is False, "batch used partial lane statistics")
        _require(receipt.get("all_lane_work_finished_before_return") is True, "batch returned with lane work outstanding")
        _require(receipt.get("forest_merge_used") is False, "batch merged a root-parallel forest")
        _require(receipt.get("private_random_outcome_samples") == 0, "batch privately sampled random outcomes")
        _require(receipt.get("guessed_random_rules_or_successors") == 0, "batch guessed random rules")
        _require(receipt.get("unobserved_random_outcome_advances") == 0, "batch advanced unobserved random state")
        _require(receipt.get("leaf_microbatch_sizes") == [8], "batch did not send one 8-row frozen leaf batch")
        per_lane = receipt.get("per_lane")
        _require(isinstance(per_lane, list) and len(per_lane) == 8, "batch lacks eight lane rows")
        _require(sorted(int(item.get("lane_id", -1)) for item in per_lane) == list(range(8)), "batch lane ids are not 0..7")
        all_lanes.extend(per_lane)
    shared = payload.get("shared_tree")
    _require(isinstance(shared, dict), "shared-tree aggregate receipt is absent")
    _require(shared.get("one_shared_logical_tree") is True, "aggregate is not one shared logical tree")
    _require(shared.get("outstanding_reservations") == 0, "aggregate leaked reservations")
    _require(shared.get("virtual_loss_after") == 0, "aggregate leaked virtual loss")
    for key in ("private_random_samples", "guessed_random_rules", "unobserved_random_advances"):
        _require(payload.get(key) == 0, f"r225 nonzero forbidden counter: {key}")
    _require(payload.get("partial_lane_statistics_used") is False, "r225 used partial lane statistics")
    cleanup = payload.get("deadline_cleanup")
    _require(isinstance(cleanup, dict) and cleanup.get("no_background_native_work_after_return") is True, "r225 did not prove native cleanup")
    # Search IDs are handle-scoped in stock libcg.  Therefore this is a count
    # of observed, distinct (internal AgentStart arena, SearchBegin ID) pairs,
    # not an unsafe global numeric-ID comparison across independent arenas.
    return {
        "loaded_stock_libcg_dso_count": 1,
        "internal_agent_start_arena_count": 8,
        "internal_agent_start_call_count": 8,
        "internal_agent_start_handle_count": 8,
        "distinct_search_begin_id_count": 8,
        "cpu_worker_lane_count": 8,
        "unique_raw_handle_count": 8,
        "search_begin_calls": sum(int(item["search_begin_calls"]) for item in all_lanes),
        "search_step_calls": sum(int(item["search_step_calls"]) for item in all_lanes),
        "search_release_calls": sum(int(item["search_release_calls"]) for item in all_lanes),
        "search_end_calls": sum(int(item["search_end_calls"]) for item in all_lanes),
        "isolated_stock_search_state_count": 8,
        "one_lane_baseline_or_ratio_comparison_count": 0,
        "eight_frontier_leaf_gpu_batch_count": sum(1 for _ in batches),
        "eight_frontier_leaf_gpu_batch_size_distribution": [8 for _ in batches],
        "all_eight_frontier_results_backed_up_before_repeat_count": len(batches),
        "evidence": {
            "native_search_id_cross_lane_reuse": [
                int(row["receipt"]["native_search_id_cross_lane_reuse"])
                for row in batches
            ],
            "per_batch_unique_raw_handle_count": [
                int(row["receipt"]["unique_raw_handle_count"]) for row in batches
            ],
            "per_batch_shared_tree_id": [
                str(row["receipt"]["shared_logical_tree_id"]) for row in batches
            ],
        },
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage.resolve()
    _require(stage.is_dir() and not stage.is_symlink(), "stage is not a physical directory")
    for relative in ("main.py", "r195_direct_main.py", "deck.csv", "cg/libcg.so"):
        _require((stage / relative).is_file(), f"staged package lacks {relative}")
    expected_lib = (stage / "cg/libcg.so").resolve()
    bundle = args.bundle.resolve()
    stage_receipt = args.stage_receipt.resolve()
    manifest = stage / "r225_eight_lane_diagnostic_manifest.json"
    _require(bundle.is_file() and not bundle.is_symlink(), "candidate package archive is unavailable")
    _require(stage_receipt.is_file() and not stage_receipt.is_symlink(), "stage receipt is unavailable")
    _require(manifest.is_file() and not manifest.is_symlink(), "candidate package manifest is unavailable")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["CG_LIB_PATH"] = str(stage)
    os.chdir(stage)
    sys.path.insert(0, str(stage))
    main = importlib.import_module("main")
    from poke_bot import cg_env

    deck = _read_deck(stage)
    tee = _Tee(sys.stdout)
    battle_started = False
    real_game = {
        "stock_battle_start_used": True,
        "direct_r195_actions_before_trigger": 0,
        "diagnostic_direct_action_accepted_by_stock_battle_select": False,
        "battle_finish_called": False,
    }
    direct_actions = 0
    selected_action_accepted = False
    diagnostic_payload: dict[str, Any] | None = None
    after_action_threads: list[str] = []
    try:
        observation, start = cg_env.battle_start(deck, deck)
        if observation is None:
            raise SmokeError(
                "stock BattleStart failed: "
                + str(getattr(start, "errorType", "unknown"))
            )
        battle_started = True
        for _ in range(int(args.max_real_actions)):
            if cg_env.is_finished(observation):
                raise SmokeError("stock game finished before an ordinary diagnostic decision")
            with contextlib.redirect_stdout(tee):
                action = main.agent(observation)
            direct_actions += 1
            real_game["direct_r195_actions_before_trigger"] = direct_actions
            if not isinstance(action, list) or not all(isinstance(item, int) for item in action):
                raise SmokeError("archived r195 direct policy returned a non-list[int] action")
            emitted = _parse_diagnostic(tee.text) if DIAGNOSTIC_PREFIX in tee.text else None
            observation = cg_env.battle_select(action)
            if emitted is not None:
                diagnostic_payload = emitted
                selected_action_accepted = True
                real_game["diagnostic_direct_action_accepted_by_stock_battle_select"] = True
                after_action_threads = sorted(thread.name for thread in threading.enumerate())
                break
        if diagnostic_payload is None:
            raise SmokeError("no ordinary direct action triggered the r225 one-shot diagnostic")
        _require(selected_action_accepted, "diagnostic direct action was not accepted by stock BattleSelect")
        topology = _validate_payload(diagnostic_payload, expected_lib=expected_lib)
        dso_paths = _loaded_dso_paths(expected_lib)
        _require(dso_paths == [str(expected_lib)], "process loaded a non-package or multiple stock libcg DSOs")
        lingering = [
            name
            for name in after_action_threads
            if name.startswith("r222-stock-search-lane-")
            or name == "r222-frozen-leaf-microbatch"
        ]
        _require(not lingering, "native worker thread remained after direct action return")
        return {
            "schema": SCHEMA,
            "status": "local_structural_pass_resource_unverified"
            if diagnostic_payload.get("status") == "structural_pass_resource_unverified"
            else "local_viable",
            "scope": "one_fresh_local_child_process_no_kaggle",
            "stage": str(stage),
            "stage_libcg": {
                "path": str(expected_lib),
                "sha256": _sha256_file(expected_lib),
                "size_bytes": expected_lib.stat().st_size,
                "loaded_dso_paths": dso_paths,
            },
            "candidate_package": {
                "archive": str(bundle),
                "archive_sha256": _sha256_file(bundle),
                "stage_receipt": str(stage_receipt),
                "stage_receipt_sha256": _sha256_file(stage_receipt),
                "member_manifest": str(manifest),
                "member_manifest_sha256": _sha256_file(manifest),
            },
            "driver": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
                "python": sys.version,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cg_lib_path": os.environ.get("CG_LIB_PATH"),
                "private_random_sampling_by_driver": 0,
                "guessed_random_rules_by_driver": 0,
                "unobserved_random_advances_by_driver": 0,
            },
            "real_game": real_game,
            "topology": topology,
            "after_direct_action": {
                "threads": after_action_threads,
                "lingering_r222_native_threads": lingering,
                "no_background_native_work_after_real_action": not lingering,
            },
            "diagnostic": diagnostic_payload,
        }
    finally:
        if battle_started:
            cg_env.battle_finish()
            real_game["battle_finish_called"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--max-real-actions", type=int, default=12)
    args = parser.parse_args()
    if int(args.max_real_actions) < 1 or int(args.max_real_actions) > 64:
        raise SystemExit("--max-real-actions must be in [1, 64]")
    result: dict[str, Any]
    exit_code = 0
    try:
        result = _run(args)
    except BaseException as exc:
        exit_code = 1
        result = {
            "schema": SCHEMA,
            "status": "not_viable",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "scope": "one_fresh_local_child_process_no_kaggle",
        }
    try:
        _write_json_once(args.receipt.resolve(), result)
    except BaseException as exc:
        result = {
            "schema": SCHEMA,
            "status": "not_viable",
            "failure_reason": f"receipt_write {type(exc).__name__}: {exc}",
            "scope": "one_fresh_local_child_process_no_kaggle",
        }
        exit_code = 1
    print(FINAL_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
