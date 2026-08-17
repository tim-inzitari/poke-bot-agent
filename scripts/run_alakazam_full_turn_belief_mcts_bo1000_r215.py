#!/usr/bin/env python3.11
"""Stage or run the r216 local approximate frozen-r195 BeliefMCTS mirror.

Despite the historical filename, this isolated launcher follows the r216
local-exploratory contract.  It does not import a Kaggle client, queue,
submission builder, selector, trainer, or promotion path.  A two-game canary
must prove a genuine MCTS decision before the same frozen runtime can be used
for the 500-pair BO1000.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r215_bo1000_launch import (
    R215LaunchError,
    append_progress,
    assert_python311,
    build_launch_plan,
    clean_r215_runtime_environment,
    controller_execute_request,
    make_worker_request,
    materialize_templates,
    parse_schedule,
    preflight_runtime,
    validate_approximate_canary_results,
    verify_plan_source_files,
)
from poke_bot.seeded_mirror_harness import validate_pair_first_player_seal


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R215LaunchError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise R215LaunchError(f"{label} must be a JSON object: {path}")
    return payload


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise R215LaunchError(f"refusing to overwrite differing immutable receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_bytes(body)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != body:
            raise R215LaunchError(f"receipt race wrote different bytes: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_live_summary(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically update the non-authoritative dashboard status document.

    Per-game results, launch plans, preflight, and acceptance receipts remain
    create-once.  This one small file is intentionally live so a dashboard can
    show a long BO1000's progress without parsing every immutable result.
    """

    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_bytes(body)
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _event(plan: Mapping[str, Any], kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema": "poke_bot.alakazam_local_approximate_belief_mcts_bo1000_r216_launch/v1",
        "kind": kind,
        "evaluation_id": plan.get("evaluation_id"),
        "source_identity_sha256": plan.get("source", {}).get("identity_sha256"),
        "output_identity_sha256": plan.get("output", {}).get("identity_sha256"),
        "submission_authority": False,
        "kaggle_authority": False,
        "training_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
        **fields,
    }


def _worker_command(*, request_path: Path, result_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]


def _run_worker(
    *,
    request: Mapping[str, Any],
    request_path: Path,
    result_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    _write_create_once(request_path, request)
    completed = subprocess.run(
        _worker_command(request_path=request_path, result_path=result_path),
        cwd=ROOT,
        env=dict(environment),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1200:]
        raise R215LaunchError(
            f"fresh r216 worker failed exit={completed.returncode}: {detail}"
        )
    return _read_object(result_path, label="fresh r216 worker result")


def _group_pairs(plan: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in parse_schedule(plan):
        grouped[game.pair_index].append(game.as_payload())
    pairs = [sorted(grouped[index], key=lambda item: int(item["game_index"])) for index in sorted(grouped)]
    if any(len(pair) != 2 for pair in pairs):
        raise R215LaunchError("seeded schedule has a non-two-game pair")
    return pairs


def _result_totals(game_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return dashboard-safe aggregates without interpreting a partial game."""

    experimental = {"wins": 0, "draws": 0, "losses": 0}
    control = {"wins": 0, "draws": 0, "losses": 0}
    fallbacks = 0
    simulations = 0
    genuine_mcts_turns = 0
    depth_total = 0
    depth_count = 0
    max_depth = 0
    for result in game_results:
        experimental_seat = result.get("experimental_seat")
        winner = result.get("winner_seat")
        if result.get("terminal_status") == "completed" and experimental_seat in {0, 1}:
            if winner is None:
                experimental["draws"] += 1
                control["draws"] += 1
            elif winner == experimental_seat:
                experimental["wins"] += 1
                control["losses"] += 1
            elif winner in {0, 1}:
                experimental["losses"] += 1
                control["wins"] += 1
        rows = result.get("experimental_turn_receipts", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fallbacks += int(bool(row.get("direct_policy_fallback_used")))
            sims = int(row.get("sims_run", 0) or 0)
            simulations += max(0, sims)
            if sims >= 1 and not bool(row.get("direct_policy_fallback_used")):
                genuine_mcts_turns += 1
            depth = max(0, int(row.get("max_simulator_search_depth", 0) or 0))
            depth_total += depth
            depth_count += 1
            max_depth = max(max_depth, depth)
    return {
        "experimental": experimental,
        "control": control,
        "fallbacks": fallbacks,
        "simulations": simulations,
        "genuine_mcts_turns": genuine_mcts_turns,
        "average_depth": (depth_total / depth_count) if depth_count else 0.0,
        "max_depth": max_depth,
    }


def _summary_payload(
    *,
    plan: Mapping[str, Any],
    output_dir: Path,
    tracker_progress: Path,
    game_results: Sequence[Mapping[str, Any]],
    status: str,
    started_at: str,
    active_workers: int,
    error: str | None = None,
) -> dict[str, Any]:
    output = plan.get("output")
    output_map = output if isinstance(output, Mapping) else {}
    total_games = int(output_map.get("game_count", 0) or 0)
    totals = _result_totals(game_results)
    return {
        "schema": "poke_bot.alakazam_local_approximate_belief_mcts_bo1000_r216_summary/v1",
        "evaluation_id": plan.get("evaluation_id"),
        "mode": plan.get("mode"),
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "source_identity_sha256": plan.get("source", {}).get("identity_sha256"),
        "output_identity_sha256": output_map.get("identity_sha256"),
        "content_addressed_output_dir": str(output_dir),
        "progress_jsonl": str(tracker_progress),
        "completed_games": len(game_results),
        "total_games": total_games,
        "active_workers": int(active_workers),
        "experimental": totals["experimental"],
        "control": totals["control"],
        "fallbacks": totals["fallbacks"],
        "simulations": totals["simulations"],
        "genuine_mcts_turns": totals["genuine_mcts_turns"],
        "average_depth": totals["average_depth"],
        "max_depth": totals["max_depth"],
        "labels": {
            "local_approximate": True,
            "root_sampled_non_r207_exact_chance": True,
            "frozen_r195_both_arms": True,
            "matchup_adapter_enabled": True,
            "rtp_enabled": False,
            "guide_linear_enabled": False,
            "guide_logit_enabled": False,
            "guide2vec_enabled": False,
            "evaluation_only": True,
            "submission_authority": False,
            "external_submission_authority": False,
            "training_authority": False,
            "selector_authority": False,
            "promotion_authority": False,
        },
        "error": error,
    }


def _append_progress_paths(paths: Sequence[Path], event: Mapping[str, Any]) -> None:
    for path in paths:
        append_progress(path, event)


def _execute(args: argparse.Namespace) -> int:
    plan = _read_object(args.plan.resolve(), label="r216 launch plan")
    verify_plan_source_files(plan=plan, repo_root=ROOT)
    preflight = preflight_runtime(
        plan=plan,
        repo_root=ROOT,
        bundle=args.r195_bundle.resolve(),
        package_root=args.frozen_package_root.resolve(),
        seeded_engine_lib=args.seeded_engine_lib.resolve(),
        prerequisite_receipts=(
            args.advanced_prerequisite_receipts.resolve()
            if args.advanced_prerequisite_receipts is not None
            else None
        ),
        noninterference_receipt=args.noninterference_receipt.resolve(),
        canary_acceptance=(
            args.canary_acceptance.resolve() if args.canary_acceptance is not None else None
        ),
        exploratory_local_override=args.local_exploratory_override,
    )
    paths = materialize_templates(stage_root=args.output_root.resolve(), plan=plan)
    output_dir = paths["output_dir"]
    content_progress_path = paths["progress_jsonl"]
    tracker_progress_path = args.output_root.resolve() / "progress.jsonl"
    progress_paths = (content_progress_path, tracker_progress_path)
    summary_path = args.output_root.resolve() / "summary.json"
    started_at = _utc_now()
    _write_create_once(
        args.output_root.resolve() / "output-index.json",
        {
            "schema": "poke_bot.alakazam_local_approximate_belief_mcts_bo1000_r216_output_index/v1",
            "source_identity_sha256": plan.get("source", {}).get("identity_sha256"),
            "output_identity_sha256": plan.get("output", {}).get("identity_sha256"),
            "content_addressed_output_dir": str(output_dir),
            "content_addressed_progress_jsonl": str(content_progress_path),
            "tracker_progress_jsonl": str(tracker_progress_path),
            "submission_authority": False,
            "external_submission_authority": False,
        },
    )
    environment, scrubbed = clean_r215_runtime_environment(
        package_root=args.frozen_package_root.resolve(),
        seeded_engine_lib=args.seeded_engine_lib.resolve(),
    )
    environment["PYTHONPATH"] = str(ROOT)
    _write_create_once(output_dir / "runtime-preflight.json", preflight)
    game_results: list[dict[str, Any]] = []
    _write_live_summary(
        summary_path,
        _summary_payload(
            plan=plan,
            output_dir=output_dir,
            tracker_progress=tracker_progress_path,
            game_results=game_results,
            status="starting",
            started_at=started_at,
            active_workers=0,
        ),
    )
    _append_progress_paths(
        progress_paths,
        _event(
            plan,
            "launch_started",
            mode=plan.get("mode"),
            output_dir=str(output_dir),
            sanitized_environment_scrubbed_keys=scrubbed,
        ),
    )
    for pair in _group_pairs(plan):
        pair_id = str(pair[0]["pair_id"])
        pair_dir = output_dir / "pairs" / pair_id
        seal_request = make_worker_request(
            operation="seal_pair",
            plan=plan,
            runtime_identity=preflight["frozen_runtime"],
            seeded_engine_identity=preflight["seeded_engine"],
            package_root=args.frozen_package_root.resolve(),
            pair=pair,
        )
        _write_live_summary(
            summary_path,
            _summary_payload(
                plan=plan,
                output_dir=output_dir,
                tracker_progress=tracker_progress_path,
                game_results=game_results,
                status="sealing_pair",
                started_at=started_at,
                active_workers=1,
            ),
        )
        _append_progress_paths(
            progress_paths, _event(plan, "pair_seal_dispatched", pair_id=pair_id)
        )
        seal_result = _run_worker(
            request=seal_request,
            request_path=pair_dir / "seal-request.json",
            result_path=pair_dir / "seal-result.json",
            environment=environment,
        )
        from poke_bot.r215_bo1000_launch import pair_seal_from_controller_result

        seal = pair_seal_from_controller_result(seal_result)
        from poke_bot.seeded_mirror_harness import SeededMirrorGameSpec

        validate_pair_first_player_seal(
            tuple(SeededMirrorGameSpec(**dict(item)) for item in pair), seal
        )
        _append_progress_paths(
            progress_paths,
            _event(
                plan,
                "pair_sealed",
                pair_id=pair_id,
                pair_first_player_seal_sha256=seal.identity_sha256,
                first_player_seat=seal.first_player_seat,
            ),
        )
        for game in pair:
            nonce = str(game["game_nonce_sha256"])[7:19]
            request = make_worker_request(
                operation="run_game",
                plan=plan,
                runtime_identity=preflight["frozen_runtime"],
                seeded_engine_identity=preflight["seeded_engine"],
                package_root=args.frozen_package_root.resolve(),
                game=game,
                pair_first_player_seal=seal_result,
            )
            _write_live_summary(
                summary_path,
                _summary_payload(
                    plan=plan,
                    output_dir=output_dir,
                    tracker_progress=tracker_progress_path,
                    game_results=game_results,
                    status="running",
                    started_at=started_at,
                    active_workers=1,
                ),
            )
            _append_progress_paths(
                progress_paths,
                _event(plan, "game_dispatched", pair_id=pair_id, game_nonce_sha256=game["game_nonce_sha256"]),
            )
            result = _run_worker(
                request=request,
                request_path=pair_dir / f"game-{nonce}-request.json",
                result_path=pair_dir / f"game-{nonce}-result.json",
                environment=environment,
            )
            game_results.append(result)
            _write_live_summary(
                summary_path,
                _summary_payload(
                    plan=plan,
                    output_dir=output_dir,
                    tracker_progress=tracker_progress_path,
                    game_results=game_results,
                    status="running",
                    started_at=started_at,
                    active_workers=0,
                ),
            )
            _append_progress_paths(
                progress_paths,
                _event(
                    plan,
                    "game_completed",
                    pair_id=pair_id,
                    game_nonce_sha256=game["game_nonce_sha256"],
                    terminal_status=result.get("terminal_status"),
                    winner_seat=result.get("winner_seat"),
                    direct_policy_fallback_turn_count=sum(
                        int(bool(row.get("direct_policy_fallback_used")))
                        for row in result.get("experimental_turn_receipts", [])
                        if isinstance(row, Mapping)
                    ),
                ),
            )
    if plan.get("mode") == "canary":
        acceptance = validate_approximate_canary_results(
            plan=plan,
            game_results=game_results,
            frozen_runtime=preflight["frozen_runtime"],
            controller=preflight["controller"],
        )
        acceptance_path = output_dir / "canary-acceptance.json"
        _write_create_once(acceptance_path, acceptance)
        _append_progress_paths(
            progress_paths,
            _event(
                plan,
                "canary_accepted",
                canary_acceptance_path=str(acceptance_path),
                genuine_mcts_turn_count=acceptance["genuine_mcts_turn_count"],
            ),
        )
    else:
        summary = {
            "schema": "poke_bot.alakazam_local_approximate_belief_mcts_bo1000_r216_launch/v1",
            "status": "completed_all_1000_games_no_early_stop",
            "evaluation_id": plan.get("evaluation_id"),
            "game_count": len(game_results),
            "source_identity_sha256": plan.get("source", {}).get("identity_sha256"),
            "output_identity_sha256": plan.get("output", {}).get("identity_sha256"),
            "progress_jsonl": str(tracker_progress_path),
            "submission_authority": False,
            "kaggle_authority": False,
            "training_authority": False,
            "selector_authority": False,
            "promotion_authority": False,
        }
        _write_create_once(output_dir / "bo1000-completion.json", summary)
        _append_progress_paths(
            progress_paths, _event(plan, "bo1000_completed", game_count=len(game_results))
        )
    _write_live_summary(
        summary_path,
        _summary_payload(
            plan=plan,
            output_dir=output_dir,
            tracker_progress=tracker_progress_path,
            game_results=game_results,
            status=("accepted_canary" if plan.get("mode") == "canary" else "completed"),
            started_at=started_at,
            active_workers=0,
        ),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "progress_jsonl": str(tracker_progress_path),
                "game_count": len(game_results),
                "mode": plan.get("mode"),
                "kaggle_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    plan = build_launch_plan(
        repo_root=ROOT,
        mode=args.mode,
        canary_pairs=args.canary_pairs,
    )
    paths = materialize_templates(stage_root=args.stage_root.resolve(), plan=plan)
    print(
        json.dumps(
            {
                "launch_plan": str(paths["launch_plan"]),
                "source_dir": str(paths["source_dir"]),
                "output_dir": str(paths["output_dir"]),
                "progress_jsonl": str(paths["progress_jsonl"]),
                "launch_blockers": plan["launch_blockers"],
                "local_exploratory_bo1000_authorized": True,
                "kaggle_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _worker(args: argparse.Namespace) -> int:
    request = _read_object(args.request.resolve(), label="r216 worker request")
    result = controller_execute_request(request)
    _write_create_once(args.result.resolve(), result)
    print(json.dumps({"result": str(args.result.resolve()), "operation": result["operation"]}, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--mode", choices=("canary", "bo1000"), default="bo1000")
    parser.add_argument("--canary-pairs", type=int)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--r195-bundle", type=Path)
    parser.add_argument("--frozen-package-root", type=Path)
    parser.add_argument("--seeded-engine-lib", type=Path)
    parser.add_argument("--noninterference-receipt", type=Path)
    parser.add_argument("--canary-acceptance", type=Path)
    parser.add_argument("--advanced-prerequisite-receipts", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--local-exploratory-override", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.stage_root is None:
            parser.error("--prepare requires --stage-root")
        if args.mode == "bo1000" and args.canary_pairs is not None:
            parser.error("--canary-pairs is only meaningful with --mode canary")
    elif args.worker:
        if args.request is None or args.result is None:
            parser.error("--worker requires --request and --result")
    else:
        required = {
            "--plan": args.plan,
            "--r195-bundle": args.r195_bundle,
            "--frozen-package-root": args.frozen_package_root,
            "--seeded-engine-lib": args.seeded_engine_lib,
            "--noninterference-receipt": args.noninterference_receipt,
            "--output-root": args.output_root,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--execute requires " + ", ".join(missing))
        if not args.local_exploratory_override:
            parser.error("--execute requires --local-exploratory-override")
    return args


def main() -> int:
    assert_python311()
    args = _parse_args()
    if args.prepare:
        return _prepare(args)
    if args.worker:
        return _worker(args)
    return _execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R215LaunchError as exc:
        print(f"r216 local approximate BeliefMCTS launcher failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
