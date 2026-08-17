#!/usr/bin/env python3
"""Prepare or run the sealed local r219 BeliefMCTS mirror.

The only executable evaluation path is a fresh child process rooted at an
immutable r219 source snapshot.  Both arms use the frozen r195 NO-RTP model,
deck, and Matchup Adapter tree; the experimental arm uses the static r219
shared-turn controller.  This script has no Kaggle, training, selector,
service, RTP, guide, or promotion path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1"
GAME_SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219_game/v1"
EVALUATION_ID = "alakazam-r219-local-multi-search-turn-belief-mcts-bo1000"
CANARY_EVALUATION_ID = "alakazam-r219-local-multi-search-turn-belief-mcts-canary10"
OWNER_DECISION_REVISION = 219
TOTAL_PAIRS = 500
TOTAL_GAMES = TOTAL_PAIRS * 2
CANARY_PAIRS = 5
CANARY_GAMES = CANARY_PAIRS * 2
GAME_SECONDS = 600.0
GAME_RESERVE_SECONDS = 30.0
TURN_POOL_SECONDS = 45.0
TURN_POOL_DIVISOR = 8.0
SEARCH_SEGMENT_SECONDS = 15.0
EMERGENCY_SIMULATION_SAFETY_CEILING = 1_000_000
CHILD_TIMEOUT_SECONDS = 900.0
BLACKWELL_GPU_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
B77_ENGINE_SHA256 = (
    "sha256:b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6"
)
R215_CONTROLLER_SHA256 = (
    "sha256:e3ead0a14e0c56d53343829e10f2c6e6452d64c0df6973ad3d56fa291c5ac9ac"
)
R219_CONTROLLER_SHA256 = (
    "sha256:af97bafeea18044a879d2b15d41aca506eacb4ad5985ed3ac910c4ad1b993db6"
)
BELIEF_MCTS_SHA256 = (
    "sha256:c0b905a88c68675ba3b4c2f12a2425a13f0e9a61288fa6309d829340e55b4afd"
)
R219_CONTRACT_SHA256 = (
    "sha256:0ba3e67de761eae8c189cf4bf9900ff01574b54941ca42d0dbdc2b9fdb134f3e"
)
R219_CONTRACT_RELATIVE = Path(
    "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json"
)
RUNNER_RELATIVE = Path(
    "scripts/run_alakazam_local_multi_search_turn_belief_mcts_bo1000_r219.py"
)
SOURCE_MANIFEST_NAME = "r219-source-manifest.json"


class R219RunnerError(RuntimeError):
    """The local-only r219 evaluation boundary is incomplete or unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R219RunnerError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise R219RunnerError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    partial.write_bytes(encoded)
    os.replace(partial, path)


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _physical_directory(path: Path, *, label: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise R219RunnerError(f"cannot stat {label}: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise R219RunnerError(f"{label} must be a physical directory: {path}")
    return path.resolve()


def _physical_file(path: Path, *, label: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise R219RunnerError(f"cannot stat {label}: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise R219RunnerError(f"{label} must be a physical regular file: {path}")
    return path.resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _deck_cards_sha256(path: Path) -> str:
    cards: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise R219RunnerError(f"cannot read frozen r195 deck: {path}") from exc
    for source in lines:
        line = source.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise R219RunnerError("frozen r195 deck contains a noninteger card") from exc
    if len(cards) != 60:
        raise R219RunnerError(f"frozen r195 deck must have 60 cards, got {len(cards)}")
    return _canonical_sha256(cards)


def _derive_seed(evaluation_id: str, pair_index: int, label: str) -> int:
    digest = hashlib.sha256(
        f"{evaluation_id}:pair:{pair_index}:{label}".encode("utf-8")
    ).digest()
    return (int.from_bytes(digest[:4], "big") % 0xFFFFFFFF) + 1


def _validate_source(args: argparse.Namespace) -> dict[str, Any]:
    root = _physical_directory(args.source_root, label="r219 source root")
    runner = _physical_file(root / RUNNER_RELATIVE, label="canonical r219 runner")
    if runner != Path(__file__).resolve() and not args.worker:
        # A parent is intentionally started from the sealed copy, not a mutable
        # checkout that happens to point at the sealed source through an option.
        raise R219RunnerError(
            "parent must execute the canonical runner physically inside --source-root"
        )
    manifest_path = _physical_file(root / SOURCE_MANIFEST_NAME, label="r219 source manifest")
    manifest = _read_json(manifest_path, label="r219 source manifest")
    if (
        manifest.get("schema")
        != "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219_source_snapshot/v1"
        or manifest.get("owner_decision_revision") != OWNER_DECISION_REVISION
        or manifest.get("status") != "sealed_evaluation_only_source_snapshot"
    ):
        raise R219RunnerError("r219 source manifest identity is not sealed revision 219")
    source_identity = manifest.get("source_tree_sha256")
    if not isinstance(source_identity, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", source_identity
    ):
        raise R219RunnerError("r219 source manifest has no valid source-tree digest")
    expected_files = {
        root / "poke_bot/belief_mcts.py": BELIEF_MCTS_SHA256,
        root / "poke_bot/r215_full_turn_belief_mcts.py": R215_CONTROLLER_SHA256,
        root / "poke_bot/r219_multi_search_turn_belief_mcts.py": R219_CONTROLLER_SHA256,
        root / R219_CONTRACT_RELATIVE: R219_CONTRACT_SHA256,
    }
    for path, expected in expected_files.items():
        _physical_file(path, label=f"r219 sealed source {path.relative_to(root)}")
        if _sha256(path) != expected:
            raise R219RunnerError(
                f"sealed r219 source hash drifted: {path.relative_to(root)}"
            )
    contract = _read_json(root / R219_CONTRACT_RELATIVE, label="r219 contract")
    if (
        contract.get("schema") != SCHEMA
        or contract.get("owner_decision_revision") != OWNER_DECISION_REVISION
    ):
        raise R219RunnerError("sealed r219 typed contract identity drifted")

    direct = _physical_directory(args.direct_package or root / "direct", label="direct package")
    mcts = _physical_directory(args.mcts_package or root / "mcts", label="mcts package")
    for package, name in ((direct, "direct"), (mcts, "mcts")):
        if not _inside(package, root):
            raise R219RunnerError(f"{name} package must reside inside sealed source root")
        for relative in (
            "main.py",
            "model.pt",
            "deck.csv",
            "matchup_tree.json",
            "runtime_profile.json",
            "cg/libcg.so",
        ):
            _physical_file(package / relative, label=f"{name} package {relative}")
        if _sha256(package / "model.pt") != R195_CHECKPOINT_SHA256:
            raise R219RunnerError(f"{name} package is not the exact frozen r195 model")
        if _sha256(package / "matchup_tree.json") != R195_MATCHUP_TREE_SHA256:
            raise R219RunnerError(f"{name} package lost the exact r195 matchup tree")
        if _sha256(package / "cg/libcg.so") != B77_ENGINE_SHA256:
            raise R219RunnerError(f"{name} package lost the exact seeded B77 engine")
        if (package / "rtp_shadow_planner.pt").exists():
            raise R219RunnerError(f"{name} package contains forbidden RTP sidecar")
        profile = _read_json(package / "runtime_profile.json", label=f"{name} profile")
        if (
            profile.get("display") != "NO RTP"
            or profile.get("recursive_turn_planner") != "disabled"
            or profile.get("rtp_sidecar_packaged") is not False
        ):
            raise R219RunnerError(f"{name} package is not explicit frozen NO RTP")
    if _deck_cards_sha256(direct / "deck.csv") != _deck_cards_sha256(mcts / "deck.csv"):
        raise R219RunnerError("direct/mcts frozen deck card order diverged")
    direct_config = _read_json(direct / "search_config.json", label="direct search config")
    mcts_config = _read_json(mcts / "search_config.json", label="mcts search config")
    if direct_config.get("enabled") is not False or mcts_config.get("enabled") is not True:
        raise R219RunnerError("frozen package search roles are not direct versus BeliefMCTS")
    engine = direct / "cg/libcg.so"
    return {
        "source_root": str(root),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_tree_sha256": source_identity,
        "canonical_runner_sha256": _sha256(runner),
        "controller_hashes": {
            "belief_mcts": BELIEF_MCTS_SHA256,
            "r215_full_turn": R215_CONTROLLER_SHA256,
            "r219_multi_search_turn": R219_CONTROLLER_SHA256,
        },
        "direct_package_root": str(direct),
        "mcts_package_root": str(mcts),
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "deck_cards_sha256": _deck_cards_sha256(direct / "deck.csv"),
        "seeded_engine": {
            "path": str(engine),
            "sha256": B77_ENGINE_SHA256,
            "bytes": int(engine.stat().st_size),
            "battle_start_seeded_available": True,
        },
        "mcts_enabled": True,
        "direct_search_enabled": False,
        "matchup_adapter_required_on_both_arms": True,
        "rtp_enabled": False,
        "legacy_rtp_enabled": False,
        "guide_linear_enabled": False,
        "guide_logit_enabled": False,
        "guide2vec_enabled": False,
        "kaggle_authority": False,
        "training_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
    }


def _mode_values(args: argparse.Namespace) -> tuple[str, int, int, int, int]:
    if args.mode == "canary":
        if args.pair_start != 0 or args.pair_count != CANARY_PAIRS:
            raise R219RunnerError(
                "r219 canary is exactly --pair-start 0 --pair-count 5"
            )
        return CANARY_EVALUATION_ID, CANARY_PAIRS, CANARY_GAMES, 0, CANARY_PAIRS
    if args.canary_summary is None:
        raise R219RunnerError("BO1000 requires --canary-summary from a valid r219 canary")
    if args.pair_start < 0 or args.pair_count < 1 or args.pair_start + args.pair_count > TOTAL_PAIRS:
        raise R219RunnerError("BO1000 pair range must remain within 0..499")
    return EVALUATION_ID, TOTAL_PAIRS, TOTAL_GAMES, args.pair_start, args.pair_count


def _validate_canary_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="r219 canary summary")
    required = {
        "schema": SCHEMA,
        "evaluation_id": CANARY_EVALUATION_ID,
        "mode": "canary",
        "status": "complete",
        "completed_games": CANARY_GAMES,
        "valid_games": CANARY_GAMES,
        "canary_ready_for_bo1000": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise R219RunnerError("r219 canary summary is incomplete or not launch-valid")
    return payload


def _pair_schedule(evaluation_id: str, pair_index: int) -> dict[str, Any]:
    pair_nonce = _canonical_sha256(
        {"schema": SCHEMA, "evaluation_id": evaluation_id, "pair_index": pair_index}
    )
    engine_seed = _derive_seed(evaluation_id, pair_index, "engine-and-deck-order")
    games: list[dict[str, Any]] = []
    for game_index in (0, 1):
        experimental_seat = game_index
        games.append(
            {
                "pair_id": f"{evaluation_id}:pair:{pair_index:04d}",
                "pair_index": pair_index,
                "pair_nonce_sha256": pair_nonce,
                "game_index": game_index,
                "game_nonce_sha256": _canonical_sha256(
                    {
                        "schema": GAME_SCHEMA,
                        "pair_nonce_sha256": pair_nonce,
                        "game_index": game_index,
                    }
                ),
                "engine_seed_u32": engine_seed,
                "deck_order_seed_u32": engine_seed,
                "experimental_rng_seed_u32": _derive_seed(
                    evaluation_id, pair_index, f"mcts:{game_index}"
                ),
                "control_rng_seed_u32": _derive_seed(
                    evaluation_id, pair_index, f"direct:{game_index}"
                ),
                "chance_rng_seed_u32": _derive_seed(
                    evaluation_id, pair_index, f"manual-chance:{game_index}"
                ),
                "experimental_seat": experimental_seat,
                "control_seat": 1 - experimental_seat,
            }
        )
    return {
        "pair_index": pair_index,
        "pair_nonce_sha256": pair_nonce,
        "seed": engine_seed,
        "games": games,
    }


def _activate_worker_source(source_root: Path) -> None:
    """Make worker imports resolve only from its immutable r219 source root."""

    root = _physical_directory(source_root, label="worker r219 source root")
    expected = str(root)
    if os.environ.get("R219_SOURCE_ROOT") != expected:
        raise R219RunnerError("worker R219_SOURCE_ROOT does not bind its source root")
    if os.environ.get("PYTHONPATH") != expected:
        raise R219RunnerError("worker PYTHONPATH is not only its sealed source root")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get(
        "PYTHONNOUSERSITE"
    ) != "1":
        raise R219RunnerError("worker lacks sealed Python isolation flags")
    script = Path(__file__).resolve()
    if not _inside(script, root):
        raise R219RunnerError("worker runner is not physically inside sealed source root")
    stdlib_paths = [
        item
        for item in sys.path
        if item
        and item != expected
        and not _inside(Path(item), root)
    ]
    sys.path[:] = [expected, *stdlib_paths]
    for name in list(sys.modules):
        if name == "poke_bot" or name.startswith("poke_bot."):
            del sys.modules[name]


def _worker_gpu_receipt(expected_uuid: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=15.0,
    )
    observed = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    if completed.returncode != 0 or expected_uuid not in observed:
        raise R219RunnerError(
            "worker cannot receipt the requested Blackwell UUID from nvidia-smi"
        )
    return {
        "requested_cuda_visible_devices": expected_uuid,
        "requested_nvidia_visible_devices": expected_uuid,
        "nvidia_smi_visible_uuids": observed,
        "blackwell_uuid_verified": True,
    }


def _worker(args: argparse.Namespace) -> int:
    _activate_worker_source(args.source_root)
    request = _read_json(args.request, label="worker request")
    if request.get("source_root") != str(args.source_root.resolve()):
        raise R219RunnerError("worker request source root differs from process root")
    started = _utc_now()
    expected_nonce = request.get("worker_nonce")
    if not isinstance(expected_nonce, str) or not expected_nonce.startswith("sha256:"):
        raise R219RunnerError("worker request has no immutable child nonce")
    try:
        gpu = _worker_gpu_receipt(str(request["blackwell_gpu_uuid"]))
        import importlib

        runtime = importlib.import_module("poke_bot.r219_seeded_mirror_runtime")
        result = runtime.run_r219_seeded_mirror_operation(request)
        if not isinstance(result, dict):
            raise R219RunnerError("r219 worker runtime returned no object receipt")
        result["worker_process"] = {
            "pid": os.getpid(),
            "started_at_utc": started,
            "worker_nonce": expected_nonce,
            "pair_seed_u32": request.get("pair_seed_u32"),
            "fresh_process": True,
            "gpu": gpu,
        }
        _write_json_atomic(args.result, result)
        return 0
    except Exception as exc:
        _write_json_atomic(
            args.result,
            {
                "schema": GAME_SCHEMA,
                "worker_failure": True,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "worker_process": {
                    "pid": os.getpid(),
                    "started_at_utc": started,
                    "worker_nonce": expected_nonce,
                    "pair_seed_u32": request.get("pair_seed_u32"),
                    "fresh_process": True,
                },
            },
        )
        return 2


def _child_environment(
    *, runtime: Mapping[str, Any], device: str, source_root: Path
) -> list[str]:
    engine = runtime["seeded_engine"]
    return [
        "HOME=/nonexistent",
        "LANG=C.UTF-8",
        f"PATH={os.defpath}",
        f"PYTHONPATH={source_root}",
        f"R219_SOURCE_ROOT={source_root}",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        f"CUDA_VISIBLE_DEVICES={device}",
        f"NVIDIA_VISIBLE_DEVICES={device}",
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER=0",
        "POKEBOT_RTP_ENABLED=0",
        "POKEBOT_LEGACY_RTP_ENABLED=0",
        "POKEBOT_GUIDE_LINEAR_ENABLED=0",
        "POKEBOT_GUIDE_LOGIT_ENABLED=0",
        "POKEBOT_GUIDE2VEC_ENABLED=0",
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME=1",
        f"POKEBOT_LIBCG_PATH={engine['path']}",
        f"CG_LIB_PATH={runtime['direct_package_root']}",
    ]


def _run_child(
    *,
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    request: dict[str, Any],
    result_path: Path,
    log_path: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nonce = _canonical_sha256(
        {
            "schema": SCHEMA,
            "operation": request.get("operation"),
            "pair_nonce": request.get("pair_nonce_sha256"),
            "game_nonce": (request.get("game") or {}).get("game_nonce_sha256"),
            "pid_unassigned_nonce": uuid.uuid4().hex,
        }
    )
    request["worker_nonce"] = nonce
    request["source_root"] = str(args.source_root.resolve())
    request["source_identity_sha256"] = runtime["source_tree_sha256"]
    request["output_identity_sha256"] = _canonical_sha256(
        {"source": runtime["source_tree_sha256"], "output": str(result_path)}
    )
    request["blackwell_gpu_uuid"] = device
    request_path = result_path.with_suffix(".request.json")
    _write_json_atomic(request_path, request)
    command = [
        "/usr/bin/env",
        "-i",
        *_child_environment(runtime=runtime, device=device, source_root=args.source_root.resolve()),
        sys.executable,
        str((args.source_root / RUNNER_RELATIVE).resolve()),
        "--worker",
        "--source-root",
        str(args.source_root.resolve()),
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=args.source_root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        child_pid = process.pid
        timed_out = False
        try:
            exit_code = process.wait(timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            # This is the isolated evaluator child created immediately above,
            # never an SSH/Codex/editor/interactive user process.
            process.terminate()
            try:
                exit_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10.0)
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    child = {
        "pid": child_pid,
        "started_at_utc": started_at,
        "worker_nonce": nonce,
        "pair_seed_u32": request.get("pair_seed_u32"),
        "fresh_process": True,
        "requested_blackwell_uuid": device,
        "timeout_seconds": args.child_timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "wall_seconds": elapsed,
        "command_uses_env_i": True,
    }
    if not result_path.is_file():
        return {"worker_failure": True, "failure_reason": "worker_emitted_no_receipt"}, child
    payload = _read_json(result_path, label="worker result")
    worker = payload.get("worker_process")
    if not isinstance(worker, Mapping) or worker.get("pid") != child_pid or worker.get(
        "worker_nonce"
    ) != nonce:
        return {
            "worker_failure": True,
            "failure_reason": "worker_pid_or_nonce_receipt_mismatch",
        }, child
    if timed_out:
        return {
            "worker_failure": True,
            "failure_reason": "bounded_child_timeout",
        }, child
    return payload, child


def _game_valid(document: Mapping[str, Any]) -> tuple[bool, str | None]:
    if document.get("worker_failure") is True:
        return False, str(document.get("failure_reason") or "worker_failure")
    if document.get("schema") != GAME_SCHEMA or document.get("terminal_status") != "completed":
        return False, "game_not_terminal"
    if document.get("invalid_action") is True or document.get("crash") is True:
        return False, "invalid_action_or_crash"
    rows = document.get("experimental_turn_receipts")
    closes = document.get("experimental_actual_turn_close_receipts")
    if not isinstance(rows, list) or not isinstance(closes, list):
        return False, "missing_r219_turn_telemetry"
    row_turns = {
        tuple(row.get("actual_turn_key", ()))
        for row in rows
        if isinstance(row, Mapping) and len(row.get("actual_turn_key", ())) == 2
    }
    closed_turns = {
        (close.get("seat"), close.get("actual_turn_id"))
        for close in closes
        if isinstance(close, Mapping)
    }
    if row_turns != closed_turns:
        return False, "actual_turn_close_receipts_do_not_match_decision_turns"
    if any(
        float(row.get("effective_search_segment_allowance_s", 0.0) or 0.0)
        > SEARCH_SEGMENT_SECONDS + 1e-6
        for row in rows
        if isinstance(row, Mapping) and row.get("fresh_mcts_search_executed")
    ):
        return False, "search_segment_receipt_exceeds_15_seconds"
    if any(
        float(close.get("effective_actual_turn_planner_pool_seconds", 0.0) or 0.0)
        > TURN_POOL_SECONDS + 1e-6
        for close in closes
        if isinstance(close, Mapping)
    ):
        return False, "actual_turn_pool_receipt_exceeds_45_seconds"
    return True, None


def _game_record(
    *,
    document: Mapping[str, Any],
    child: Mapping[str, Any],
    pair_index: int,
    game_index: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    valid, reason = _game_valid(document)
    rows = [
        row
        for row in document.get("experimental_turn_receipts", [])
        if isinstance(row, Mapping)
    ]
    closes = [
        close
        for close in document.get("experimental_actual_turn_close_receipts", [])
        if isinstance(close, Mapping)
    ]
    result = {
        "pair_index": pair_index,
        "game_index": game_index,
        "seed": seed,
        "mcts_seat": document.get("experimental_seat"),
        "mcts_actual_turn_order": document.get("experimental_actual_turn_order"),
        "winner_seat": document.get("winner_seat"),
        "valid": valid,
        "reason": reason,
        "physical_cuda_device": device,
        "child": dict(child),
        "mcts_turn_closures": closes,
        "mcts_rows": rows,
        "manual_chance_events": document.get("manual_chance_events", []),
        "mcts_simulations": sum(int(row.get("sims", 0) or 0) for row in rows),
        "mcts_leaf_evaluations": sum(
            int(row.get("leaf_evaluations", 0) or 0) for row in rows
        ),
        "mcts_root_visits": sum(
            int(row.get("root_visits", 0) or 0) for row in rows
        ),
        "fallbacks": sum(
            bool(row.get("direct_policy_fallback_used")) for row in rows
        ),
        "converged_searches": sum(bool(row.get("root_action_stable")) for row in rows),
        "mcts_action_changes_relative_to_frozen_direct_policy": sum(
            bool(row.get("mcts_changed_action_relative_to_frozen_direct_policy"))
            for row in rows
        ),
        "timing_breaches": sum(
            bool(row.get("timing_breach_observed")) for row in rows
        ),
        "max_depth": max((int(row.get("max_depth", 0) or 0) for row in rows), default=0),
        "search_segments": sum(
            int(close.get("search_segments_this_turn", 0) or 0) for close in closes
        ),
        "first_search_segments": sum(
            1
            for row in rows
            if row.get("fresh_mcts_search_executed")
            and int(row.get("search_segment_index", 0) or 0) == 1
        ),
        "later_search_segments": sum(
            1
            for row in rows
            if row.get("fresh_mcts_search_executed")
            and int(row.get("search_segment_index", 0) or 0) > 1
        ),
    }
    return result


def _run_pair(
    *,
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    evaluation_id: str,
    pair_index: int,
    games_dir: Path,
    logs_dir: Path,
    device: str,
) -> list[dict[str, Any]]:
    schedule = _pair_schedule(evaluation_id, pair_index)
    pair_dir = games_dir / f"pair-{pair_index:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    seal_request = {
        "operation": "seal_pair",
        "evaluation_id": evaluation_id,
        "pair_seed_u32": schedule["seed"],
        "pair_nonce_sha256": schedule["pair_nonce_sha256"],
        "package_root": runtime["direct_package_root"],
        "seeded_engine": runtime["seeded_engine"],
        "seeded_engine_lib": runtime["seeded_engine"]["path"],
        "pair": schedule["games"],
    }
    seal, seal_child = _run_child(
        args=args,
        runtime=runtime,
        request=seal_request,
        result_path=pair_dir / "pair-seal.json",
        log_path=logs_dir / f"pair-{pair_index:04d}-seal.log",
        device=device,
    )
    records: list[dict[str, Any]] = []
    if seal.get("worker_failure") or seal.get("first_player_seat") not in {0, 1}:
        reason = str(seal.get("failure_reason") or "pair_seal_failed")
        for game in schedule["games"]:
            records.append(
                {
                    "pair_index": pair_index,
                    "game_index": game["game_index"],
                    "seed": schedule["seed"],
                    "mcts_seat": game["experimental_seat"],
                    "valid": False,
                    "reason": reason,
                    "physical_cuda_device": device,
                    "child": dict(seal_child),
                    "mcts_turn_closures": [],
                    "mcts_rows": [],
                    "manual_chance_events": [],
                }
            )
        return records
    for game in schedule["games"]:
        game_request = {
            "operation": "run_game",
            "evaluation_id": evaluation_id,
            "pair_seed_u32": schedule["seed"],
            "pair_nonce_sha256": schedule["pair_nonce_sha256"],
            "direct_package_root": runtime["direct_package_root"],
            "mcts_package_root": runtime["mcts_package_root"],
            "seeded_engine": runtime["seeded_engine"],
            "seeded_engine_lib": runtime["seeded_engine"]["path"],
            "pair_first_player_seal": seal,
            "game": game,
        }
        output = pair_dir / f"game-{game['game_index']}.json"
        document, child = _run_child(
            args=args,
            runtime=runtime,
            request=game_request,
            result_path=output,
            log_path=logs_dir / f"pair-{pair_index:04d}-game-{game['game_index']}.log",
            device=device,
        )
        records.append(
            _game_record(
                document=document,
                child=child,
                pair_index=pair_index,
                game_index=int(game["game_index"]),
                seed=schedule["seed"],
                device=device,
            )
        )
    return records


def _summary(
    *,
    started: float,
    records: Sequence[Mapping[str, Any]],
    mode: str,
    evaluation_id: str,
    total_pairs: int,
    total_games: int,
    pair_start: int,
    pair_count: int,
    workers: int,
) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid") is True]
    closures = [
        close
        for record in records
        for close in record.get("mcts_turn_closures", [])
        if isinstance(close, Mapping)
    ]
    rows = [
        row
        for record in records
        for row in record.get("mcts_rows", [])
        if isinstance(row, Mapping)
    ]
    total_mcts_turns = len(closures)
    segment_counts = [
        int(close.get("search_segments_this_turn", 0) or 0) for close in closures
    ]
    mcts_wins = sum(
        record.get("winner_seat") == record.get("mcts_seat") for record in valid
    )
    direct_wins = sum(
        record.get("winner_seat") in {0, 1}
        and record.get("winner_seat") != record.get("mcts_seat")
        for record in valid
    )
    draws = len(valid) - mcts_wins - direct_wins
    elapsed = max(1e-9, time.monotonic() - started)
    shard_games = pair_count * 2
    rate = len(records) * 3600.0 / elapsed
    canary_ready = (
        mode == "canary"
        and len(records) == CANARY_GAMES
        and len(valid) == CANARY_GAMES
        and sum(record.get("mcts_seat") == 0 for record in valid) == CANARY_PAIRS
        and sum(record.get("mcts_seat") == 1 for record in valid) == CANARY_PAIRS
        and sum(
            record.get("mcts_actual_turn_order") == "first" for record in valid
        )
        == CANARY_PAIRS
        and sum(
            record.get("mcts_actual_turn_order") == "second" for record in valid
        )
        == CANARY_PAIRS
        and total_mcts_turns >= 1
        and any(bool(row.get("fresh_mcts_search_executed")) for row in rows)
        and not any(bool(row.get("timing_breach_observed")) for row in rows)
    )
    return {
        "schema": SCHEMA,
        "evaluation_id": evaluation_id,
        "mode": mode,
        "status": "running" if len(records) < shard_games else "complete",
        "owner_decision_revision": OWNER_DECISION_REVISION,
        "completed_games": len(records),
        "valid_games": len(valid),
        "invalid_games": len(records) - len(valid),
        "total_games": total_games,
        "matched_pairs": total_pairs,
        "shard_pair_start": pair_start,
        "shard_pair_count": pair_count,
        "shard_total_games": shard_games,
        "active_worker_limit": workers,
        "mcts_wins": mcts_wins,
        "direct_wins": direct_wins,
        "draws": draws,
        "mcts_as_seat_0_games": sum(record.get("mcts_seat") == 0 for record in valid),
        "mcts_as_seat_1_games": sum(record.get("mcts_seat") == 1 for record in valid),
        "mcts_actual_first_games": sum(
            record.get("mcts_actual_turn_order") == "first" for record in valid
        ),
        "mcts_actual_second_games": sum(
            record.get("mcts_actual_turn_order") == "second" for record in valid
        ),
        "total_mcts_turns": total_mcts_turns,
        "turns_with_exactly_one_search_segment": sum(
            count == 1 for count in segment_counts
        ),
        "turns_with_one_or_more_later_research_segments": sum(
            int(close.get("later_research_count_this_turn", 0) or 0) >= 1
            for close in closures
        ),
        "average_search_segments_per_turn": (
            sum(segment_counts) / total_mcts_turns if total_mcts_turns else 0.0
        ),
        "maximum_search_segments_per_turn": max(segment_counts, default=0),
        "first_search_segments": sum(
            int(record.get("first_search_segments", 0) or 0) for record in records
        ),
        "later_search_segments": sum(
            int(record.get("later_search_segments", 0) or 0) for record in records
        ),
        "cache_only_later_steps": sum(
            int(close.get("cache_only_later_steps", 0) or 0) for close in closures
        ),
        "finite_chance_enumerations": sum(
            int(close.get("finite_chance_enumeration_count_this_turn", 0) or 0)
            for close in closures
        ),
        "chance_or_information_rebuilds": sum(
            bool(row.get("chance_or_information_rebuild")) for row in rows
        ),
        "manual_unbiased_chance_events": sum(
            len(record.get("manual_chance_events", [])) for record in records
        ),
        "simulations": sum(int(record.get("mcts_simulations", 0) or 0) for record in records),
        "leaf_evaluations": sum(
            int(record.get("mcts_leaf_evaluations", 0) or 0) for record in records
        ),
        "root_visits": sum(
            int(record.get("mcts_root_visits", 0) or 0) for record in records
        ),
        "depth": {
            "maximum": max(
                (int(record.get("max_depth", 0) or 0) for record in records),
                default=0,
            ),
            "mean_over_search_rows": (
                sum(int(row.get("max_depth", 0) or 0) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        },
        "convergence": {
            "stable_root_searches": sum(
                bool(row.get("stable_root_convergence")) for row in rows
            ),
            "fully_backed_selected_actions": sum(
                bool(row.get("selected_action_fully_backed_up")) for row in rows
            ),
        },
        "direct_fallbacks": sum(
            int(record.get("fallbacks", 0) or 0) for record in records
        ),
        "mcts_action_changes_relative_to_frozen_direct_policy": sum(
            int(
                record.get(
                    "mcts_action_changes_relative_to_frozen_direct_policy", 0
                )
                or 0
            )
            for record in records
        ),
        "timing_breaches": sum(
            int(record.get("timing_breaches", 0) or 0) for record in records
        ),
        "child_timeouts": sum(
            bool((record.get("child") or {}).get("timed_out")) for record in records
        ),
        "games_per_hour": rate,
        "eta_seconds": (
            max(0, shard_games - len(records)) * 3600.0 / rate if rate > 0 else None
        ),
        "elapsed_seconds": elapsed,
        "canary_ready_for_bo1000": canary_ready,
        "labels": {
            "local_approximate_belief_mcts_non_exact": True,
            "root_sampled_belief_mcts_non_r207_exact_chance": True,
            "non_promotion_exploratory_result": True,
            "frozen_r195_no_rtp_both_arms": True,
            "matchup_adapter_runtime_on_both_arms": True,
            "rtp_and_legacy_rtp_off": True,
            "guide_linear_guide_logit_guide2vec_off": True,
            "training_eligible": False,
            "kaggle_submission_authority": False,
        },
        "updated_at_utc": _utc_now(),
    }


def _parent(args: argparse.Namespace) -> int:
    evaluation_id, total_pairs, total_games, pair_start, pair_count = _mode_values(args)
    if args.mode == "bo1000":
        _validate_canary_summary(args.canary_summary)
    runtime = _validate_source(args)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise R219RunnerError(f"refusing to reuse evaluation output root: {output_root}")
    output_root.mkdir(parents=True)
    games_dir = output_root / "games"
    logs_dir = output_root / "logs"
    games_dir.mkdir()
    logs_dir.mkdir()
    _write_json_atomic(
        output_root / "run-contract.json",
        {
            "schema": SCHEMA,
            "evaluation_id": evaluation_id,
            "mode": args.mode,
            "owner_decision_revision": OWNER_DECISION_REVISION,
            "total_games": total_games,
            "matched_pairs": total_pairs,
            "pair_start": pair_start,
            "pair_count": pair_count,
            "workers": args.workers,
            "cuda_visible_devices_weighting": args.cuda_visible_devices,
            "child_timeout_seconds": args.child_timeout_seconds,
            "runtime": runtime,
            "timing": {
                "game_seconds": GAME_SECONDS,
                "reserve_seconds": GAME_RESERVE_SECONDS,
                "dynamic_game_allowance_formula": (
                    "min(45.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
                ),
                "default_actual_turn_planner_pool_seconds": TURN_POOL_SECONDS,
                "per_meaningful_search_segment_ceiling_seconds": SEARCH_SEGMENT_SECONDS,
                "later_meaningful_searches_use_residual_pool": True,
                "direct_fallback_reserve": "min(2.0, max(0.25, 2 * direct_preview_elapsed))",
                "fixed_simulation_target": None,
                "fixed_depth_target": None,
                "emergency_simulation_safety_ceiling": EMERGENCY_SIMULATION_SAFETY_CEILING,
                "convergence_requires_legal_fully_backed_receipt": True,
            },
            "no_kaggle_submission": True,
            "no_remote_service_or_training_action": True,
            "created_at_utc": _utc_now(),
        },
    )
    devices = [
        item.strip()
        for item in args.cuda_visible_devices.split(",")
        if item.strip()
    ]
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    progress = output_root / "progress.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[list[dict[str, Any]]], int] = {}
        for pair_index in range(pair_start, pair_start + pair_count):
            future = executor.submit(
                _run_pair,
                args=args,
                runtime=runtime,
                evaluation_id=evaluation_id,
                pair_index=pair_index,
                games_dir=games_dir,
                logs_dir=logs_dir,
                device=devices[pair_index % len(devices)],
            )
            futures[future] = pair_index
        while futures:
            complete, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in complete:
                pair_index = futures.pop(future)
                try:
                    pair_records = future.result()
                except Exception as exc:
                    pair_records = [
                        {
                            "pair_index": pair_index,
                            "game_index": game_index,
                            "valid": False,
                            "reason": f"parent_pair_error:{type(exc).__name__}:{exc}",
                            "mcts_turn_closures": [],
                            "mcts_rows": [],
                            "manual_chance_events": [],
                        }
                        for game_index in (0, 1)
                    ]
                records.extend(pair_records)
                for record in pair_records:
                    _append_event(
                        progress,
                        {
                            "schema": SCHEMA,
                            "kind": "game_complete",
                            "evaluation_id": evaluation_id,
                            **record,
                        },
                    )
                _write_json_atomic(
                    output_root / "summary.json",
                    _summary(
                        started=started,
                        records=records,
                        mode=args.mode,
                        evaluation_id=evaluation_id,
                        total_pairs=total_pairs,
                        total_games=total_games,
                        pair_start=pair_start,
                        pair_count=pair_count,
                        workers=args.workers,
                    ),
                )
    summary = _summary(
        started=started,
        records=records,
        mode=args.mode,
        evaluation_id=evaluation_id,
        total_pairs=total_pairs,
        total_games=total_games,
        pair_start=pair_start,
        pair_count=pair_count,
        workers=args.workers,
    )
    summary["status"] = "complete"
    _write_json_atomic(output_root / "summary.json", summary)
    return 0 if len(records) == pair_count * 2 else 2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--mode", choices=("canary", "bo1000"), default="bo1000")
    parser.add_argument("--direct-package", type=Path)
    parser.add_argument("--mcts-package", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--canary-summary", type=Path)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=BLACKWELL_GPU_UUID)
    parser.add_argument("--child-timeout-seconds", type=float, default=CHILD_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.request is None or args.result is None:
            parser.error("--worker requires --request and --result")
        return args
    if args.output_root is None:
        parser.error("parent requires --output-root")
    if args.pair_count is None:
        args.pair_count = CANARY_PAIRS if args.mode == "canary" else TOTAL_PAIRS
    if args.workers < 1 or args.workers > 96:
        parser.error("--workers must be in 1..96")
    if args.child_timeout_seconds <= 600.0:
        parser.error("--child-timeout-seconds must be greater than 600")
    devices = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
    if not devices or any(device != BLACKWELL_GPU_UUID for device in devices):
        parser.error("all --cuda-visible-devices entries must be the bound Blackwell UUID")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R219RunnerError as exc:
        print(f"r219 local mirror refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
