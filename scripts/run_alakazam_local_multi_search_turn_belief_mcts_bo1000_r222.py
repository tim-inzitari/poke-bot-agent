#!/usr/bin/env python3
"""Run the sealed r222 stock-libcg local BeliefMCTS mirror.

This is evaluator-only code.  It starts no service and has no training,
selector, serving, RTP, guide, or Kaggle path.  A BO1000 invocation performs
one required source/stock-runtime preflight, then schedules all 500
seat-swapped pairs without treating the first five pairs as a gate.
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


SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r222/v1"
GAME_SCHEMA = (
    "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r222_game/v1"
)
PREFLIGHT_SCHEMA = (
    "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r222_preflight/v1"
)
EVALUATION_ID = "alakazam-r222-local-multi-search-turn-belief-mcts-bo1000"
OWNER_DECISION_REVISION = 222
TOTAL_PAIRS = 500
TOTAL_GAMES = TOTAL_PAIRS * 2
PREFIX_PAIR_COUNT = 5
PREFIX_GAME_COUNT = PREFIX_PAIR_COUNT * 2
GAME_SECONDS = 600.0
GAME_RESERVE_SECONDS = 30.0
TURN_POOL_SECONDS = 45.0
SEARCH_SEGMENT_SECONDS = 15.0
EMERGENCY_SIMULATION_SAFETY_CEILING = 1_000_000
EIGHT_LANE_COUNT = 8
CHILD_TIMEOUT_SECONDS = 900.0
BLACKWELL_GPU_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
STOCK_LIBCG_SHA256 = (
    "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
)
STOCK_LIBCG_BYTES = 1_342_400
BELIEF_MCTS_SHA256 = (
    "sha256:ff9607c66bf05a4ae400fb434acd4d002dd97fd9b6ed7b5eb04e5819b3fede11"
)
R215_CONTROLLER_SHA256 = (
    "sha256:e3ead0a14e0c56d53343829e10f2c6e6452d64c0df6973ad3d56fa291c5ac9ac"
)
R219_CONTROLLER_SHA256 = (
    "sha256:af97bafeea18044a879d2b15d41aca506eacb4ad5985ed3ac910c4ad1b993db6"
)
# Bound again during the final source-freeze pass.  Keeping this current while
# the source is assembled prevents a stale r222 draft contract from being
# accidentally treated as the launch contract.
R222_CONTRACT_SHA256 = (
    "sha256:8b5a19e8746b8e5f667683ad6437a2f3506aa0fbdcca8495ec2b8bbd1eebeb7e"
)
R222_CONTRACT_RELATIVE = Path(
    "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json"
)
RUNNER_RELATIVE = Path(
    "scripts/run_alakazam_local_multi_search_turn_belief_mcts_bo1000_r222.py"
)
SOURCE_MANIFEST_NAME = "r222-source-manifest.json"
REQUIRED_RUNTIME_FILES = (
    "poke_bot/belief_mcts.py",
    "poke_bot/r215_full_turn_belief_mcts.py",
    "poke_bot/r219_multi_search_turn_belief_mcts.py",
    "poke_bot/r222_multi_search_turn_belief_mcts.py",
    "poke_bot/r222_stock_shared_tree_batch.py",
    "poke_bot/r222_stock_mirror_runtime.py",
)


class R222RunnerError(RuntimeError):
    """A source, runtime, or receipt fact is missing or unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded + b"\n").hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R222RunnerError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise R222RunnerError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    partial = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    partial.write_bytes(data)
    os.replace(partial, path)


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _physical_directory(path: Path, *, label: str) -> Path:
    try:
        path.lstat()
    except OSError as exc:
        raise R222RunnerError(f"cannot stat {label}: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise R222RunnerError(f"{label} must be a physical directory: {path}")
    return path.resolve()


def _physical_file(path: Path, *, label: str) -> Path:
    try:
        path.lstat()
    except OSError as exc:
        raise R222RunnerError(f"cannot stat {label}: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise R222RunnerError(f"{label} must be a physical regular file: {path}")
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
        raise R222RunnerError(f"cannot read frozen r195 deck: {path}") from exc
    for source in lines:
        line = source.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise R222RunnerError("frozen r195 deck contains a noninteger card") from exc
    if len(cards) != 60:
        raise R222RunnerError(f"frozen r195 deck must have 60 cards, got {len(cards)}")
    return _canonical_sha256(cards)


def _derive_policy_nonce(evaluation_id: str, pair_index: int, label: str) -> int:
    digest = hashlib.sha256(
        f"{evaluation_id}:pair:{pair_index}:{label}".encode("utf-8")
    ).digest()
    return (int.from_bytes(digest[:4], "big") % 0xFFFFFFFF) + 1


def _required_manifest_hashes(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = manifest.get("selected_runtime_code_sha256")
    if not isinstance(selected, Mapping):
        raise R222RunnerError(
            "r222 source manifest lacks selected_runtime_code_sha256 mapping"
        )
    return selected


def _validate_source(args: argparse.Namespace) -> dict[str, Any]:
    root = _physical_directory(args.source_root, label="r222 source root")
    runner = _physical_file(root / RUNNER_RELATIVE, label="canonical r222 runner")
    if runner != Path(__file__).resolve() and not args.worker:
        raise R222RunnerError(
            "parent must execute the canonical runner physically inside --source-root"
        )
    manifest_path = _physical_file(root / SOURCE_MANIFEST_NAME, label="r222 source manifest")
    manifest = _read_json(manifest_path, label="r222 source manifest")
    if (
        manifest.get("schema")
        != "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r222_source_snapshot/v1"
        or manifest.get("owner_decision_revision") != OWNER_DECISION_REVISION
        or manifest.get("status") != "sealed_evaluation_only_source_snapshot"
    ):
        raise R222RunnerError("r222 source manifest identity is not sealed revision 222")
    source_identity = manifest.get("source_tree_sha256")
    if not isinstance(source_identity, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", source_identity
    ):
        raise R222RunnerError("r222 source manifest has no valid source-tree digest")
    selected = _required_manifest_hashes(manifest)
    known = {
        "poke_bot/belief_mcts.py": BELIEF_MCTS_SHA256,
        "poke_bot/r215_full_turn_belief_mcts.py": R215_CONTROLLER_SHA256,
        "poke_bot/r219_multi_search_turn_belief_mcts.py": R219_CONTROLLER_SHA256,
    }
    for relative in REQUIRED_RUNTIME_FILES:
        path = _physical_file(root / relative, label=f"r222 runtime {relative}")
        expected = selected.get(relative)
        if not isinstance(expected, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", expected
        ):
            raise R222RunnerError(f"r222 manifest lacks a selected hash for {relative}")
        if _sha256(path) != expected:
            raise R222RunnerError(f"r222 selected runtime hash drifted: {relative}")
        if relative in known and expected != known[relative]:
            raise R222RunnerError(f"r222 selected runtime is not the frozen {relative}")
    stale_runtime = root / "poke_bot/r222_seeded_mirror_runtime.py"
    if stale_runtime.exists():
        raise R222RunnerError("r222 source snapshot contains a forbidden seeded runtime")
    contract_path = _physical_file(root / R222_CONTRACT_RELATIVE, label="r222 contract")
    if _sha256(contract_path) != R222_CONTRACT_SHA256:
        raise R222RunnerError("r222 typed contract hash drifted")
    contract = _read_json(contract_path, label="r222 contract")
    if (
        contract.get("schema") != SCHEMA
        or contract.get("owner_decision_revision") != OWNER_DECISION_REVISION
    ):
        raise R222RunnerError("r222 typed contract identity drifted")
    transport = contract.get("runtime_transport")
    if not isinstance(transport, Mapping) or any(
        transport.get(key) is not expected
        for key, expected in (
            ("exact_stock_libcg_archived_in_r195_required", True),
            ("b77_allowed", False),
            ("seeded_engine_or_battle_start_seeded_allowed", False),
            ("batch_or_multi_game_custom_engine_allowed", False),
        )
    ):
        raise R222RunnerError("r222 transport contract is not stock one-game only")

    direct = _physical_directory(args.direct_package or root / "direct", label="direct package")
    mcts = _physical_directory(args.mcts_package or root / "mcts", label="mcts package")
    for package, name in ((direct, "direct"), (mcts, "mcts")):
        if not _inside(package, root):
            raise R222RunnerError(f"{name} package must reside inside sealed source root")
        for relative in (
            "main.py",
            "model.pt",
            "deck.csv",
            "matchup_tree.json",
            "runtime_profile.json",
            "turn_order_profile.json",
            "cg/libcg.so",
        ):
            _physical_file(package / relative, label=f"{name} package {relative}")
        if _sha256(package / "model.pt") != R195_CHECKPOINT_SHA256:
            raise R222RunnerError(f"{name} package is not the exact frozen r195 model")
        if _sha256(package / "matchup_tree.json") != R195_MATCHUP_TREE_SHA256:
            raise R222RunnerError(f"{name} package lost the exact r195 matchup tree")
        library = package / "cg/libcg.so"
        if _sha256(library) != STOCK_LIBCG_SHA256 or library.stat().st_size != STOCK_LIBCG_BYTES:
            raise R222RunnerError(f"{name} package lost the exact stock r195 library")
        if (package / "rtp_shadow_planner.pt").exists():
            raise R222RunnerError(f"{name} package contains a forbidden RTP sidecar")
        profile = _read_json(package / "runtime_profile.json", label=f"{name} profile")
        if (
            profile.get("display") != "NO RTP"
            or profile.get("recursive_turn_planner") != "disabled"
            or profile.get("rtp_sidecar_packaged") is not False
        ):
            raise R222RunnerError(f"{name} package is not explicit frozen NO RTP")
    if _deck_cards_sha256(direct / "deck.csv") != _deck_cards_sha256(mcts / "deck.csv"):
        raise R222RunnerError("direct/mcts frozen deck card order diverged")
    direct_config = _read_json(direct / "search_config.json", label="direct search config")
    mcts_config = _read_json(mcts / "search_config.json", label="mcts search config")
    if direct_config.get("enabled") is not False or mcts_config.get("enabled") is not True:
        raise R222RunnerError("frozen package search roles are not direct versus BeliefMCTS")
    return {
        "source_root": str(root),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_tree_sha256": source_identity,
        "canonical_runner_sha256": _sha256(runner),
        "selected_runtime_code_sha256": dict(selected),
        "direct_package_root": str(direct),
        "mcts_package_root": str(mcts),
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "bundle_sha256": R195_BUNDLE_SHA256,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "deck_cards_sha256": _deck_cards_sha256(direct / "deck.csv"),
        "stock_libcg": {
            "relative_path": "cg/libcg.so",
            "sha256": STOCK_LIBCG_SHA256,
            "bytes": STOCK_LIBCG_BYTES,
            "same_bytes_in_both_arms": True,
        },
        "r222_lane_count_requested": EIGHT_LANE_COUNT,
        "rng_pairing": "independent_unmatched",
        "training_authority": False,
        "kaggle_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
    }


def _pair_schedule(evaluation_id: str, pair_index: int) -> dict[str, Any]:
    pair_nonce = _canonical_sha256(
        {"schema": SCHEMA, "evaluation_id": evaluation_id, "pair_index": pair_index}
    )
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
                "experimental_seat": experimental_seat,
                "control_seat": 1 - experimental_seat,
                # The wrapper sets seat 0 as actual first in each independent
                # stream; seat swapping makes the MCTS actual-first/second
                # totals exactly balanced without claiming matched randomness.
                "expected_actual_first_seat": 0,
                "experimental_policy_rng_nonce_u32": _derive_policy_nonce(
                    evaluation_id, pair_index, f"mcts:{game_index}"
                ),
                "control_policy_rng_nonce_u32": _derive_policy_nonce(
                    evaluation_id, pair_index, f"direct:{game_index}"
                ),
                "stock_engine_rng_stream": {
                    "mode": "opaque_independent_unmatched",
                    "stream_nonce_sha256": _canonical_sha256(
                        {
                            "pair_nonce_sha256": pair_nonce,
                            "game_index": game_index,
                            "kind": "stock_engine_stream_identity_only",
                        }
                    ),
                    "engine_seed_supplied": False,
                    "matched_with_other_pair_game": False,
                },
            }
        )
    return {"pair_index": pair_index, "pair_nonce_sha256": pair_nonce, "games": games}


def _activate_worker_source(source_root: Path) -> None:
    root = _physical_directory(source_root, label="worker r222 source root")
    expected = str(root)
    if os.environ.get("R222_SOURCE_ROOT") != expected:
        raise R222RunnerError("worker R222_SOURCE_ROOT does not bind its source root")
    if os.environ.get("PYTHONPATH") != expected:
        raise R222RunnerError("worker PYTHONPATH is not only its sealed source root")
    if os.environ.get("POKEBOT_LIBCG_PATH") or os.environ.get("CG_LIB_PATH"):
        raise R222RunnerError("worker inherited a non-stock library override")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get(
        "PYTHONNOUSERSITE"
    ) != "1":
        raise R222RunnerError("worker lacks sealed Python isolation flags")
    script = Path(__file__).resolve()
    if not _inside(script, root):
        raise R222RunnerError("worker runner is not physically inside sealed source root")
    stdlib_paths = [
        item for item in sys.path if item and item != expected and not _inside(Path(item), root)
    ]
    sys.path[:] = [expected, *stdlib_paths]
    for name in list(sys.modules):
        if name == "poke_bot" or name.startswith("poke_bot.") or name == "cg" or name.startswith("cg."):
            del sys.modules[name]


def _worker_gpu_receipt(expected_uuid: str) -> dict[str, Any]:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != expected_uuid
    ):
        raise R222RunnerError("worker GPU visibility is not bound to the Blackwell UUID")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=15.0,
    )
    observed = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or expected_uuid not in observed:
        raise R222RunnerError("worker cannot receipt the requested Blackwell UUID")
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
        raise R222RunnerError("worker request source root differs from process root")
    started = _utc_now()
    nonce = request.get("worker_nonce")
    if not isinstance(nonce, str) or not nonce.startswith("sha256:"):
        raise R222RunnerError("worker request has no immutable child nonce")
    try:
        gpu = _worker_gpu_receipt(str(request["blackwell_gpu_uuid"]))
        import importlib

        runtime = importlib.import_module("poke_bot.r222_stock_mirror_runtime")
        result = runtime.run_r222_stock_mirror_operation(request)
        if not isinstance(result, dict):
            raise R222RunnerError("r222 worker runtime returned no object receipt")
        result["worker_process"] = {
            "pid": os.getpid(),
            "started_at_utc": started,
            "worker_nonce": nonce,
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
                    "worker_nonce": nonce,
                    "fresh_process": True,
                },
            },
        )
        return 2


def _child_environment(*, device: str, source_root: Path) -> list[str]:
    return [
        "HOME=/nonexistent",
        "LANG=C.UTF-8",
        f"PATH={os.defpath}",
        f"PYTHONPATH={source_root}",
        f"R222_SOURCE_ROOT={source_root}",
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
    request.update(
        {
            "worker_nonce": nonce,
            "source_root": str(args.source_root.resolve()),
            "source_identity_sha256": runtime["source_tree_sha256"],
            "output_identity_sha256": _canonical_sha256(
                {"source": runtime["source_tree_sha256"], "output": str(result_path)}
            ),
            "blackwell_gpu_uuid": device,
            "stock_libcg": runtime["stock_libcg"],
            "lane_count_requested": EIGHT_LANE_COUNT,
        }
    )
    request_path = result_path.with_suffix(".request.json")
    _write_json_atomic(request_path, request)
    command = [
        "/usr/bin/env",
        "-i",
        *_child_environment(device=device, source_root=args.source_root.resolve()),
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
        process = subprocess.Popen(command, cwd=args.source_root, stdout=log, stderr=subprocess.STDOUT)
        child_pid = process.pid
        timed_out = False
        try:
            exit_code = process.wait(timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            # This is only the evaluator child created immediately above.
            process.terminate()
            try:
                exit_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10.0)
    child = {
        "pid": child_pid,
        "started_at_utc": started_at,
        "worker_nonce": nonce,
        "fresh_process": True,
        "requested_blackwell_uuid": device,
        "timeout_seconds": args.child_timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "wall_seconds": max(0.0, time.monotonic() - started_monotonic),
        "command_uses_env_i": True,
    }
    if not result_path.is_file():
        return {"worker_failure": True, "failure_reason": "worker_emitted_no_receipt"}, child
    payload = _read_json(result_path, label="worker result")
    worker = payload.get("worker_process")
    if not isinstance(worker, Mapping) or worker.get("pid") != child_pid or worker.get("worker_nonce") != nonce:
        return {"worker_failure": True, "failure_reason": "worker_pid_or_nonce_receipt_mismatch"}, child
    if timed_out:
        return {"worker_failure": True, "failure_reason": "bounded_child_timeout"}, child
    return payload, child


def _nonnegative_int(value: Any) -> int | None:
    """Return a receipt counter only when it is an honest integer count."""

    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _shared_tree_lane_receipt_valid(
    lane: Mapping[str, Any], *, preflight: bool
) -> tuple[bool, str | None]:
    """Validate the non-negotiable r222 one-tree/eight-lane receipt.

    The runtime adapts the native shared-tree core's internal counters into
    these stable report fields.  Missing values are not treated as zeros: the
    r222 contract requires telemetry without imputation.
    """

    if (
        lane.get("requested_lane_count") != EIGHT_LANE_COUNT
        or lane.get("active_lane_count") != EIGHT_LANE_COUNT
        or lane.get("isolated_stock_search_state_count") != EIGHT_LANE_COUNT
    ):
        return False, "eight_isolated_lanes_not_receipted"
    if lane.get("all_lanes_isolated") is not True:
        return False, "stock_search_lane_isolation_unproven"
    if lane.get("all_lanes_multistep_capable") is not True:
        return False, "stock_search_multistep_capability_unproven"
    if lane.get("shared_logical_tree") is not True:
        return False, "one_shared_logical_tree_unproven"
    if lane.get("independent_root_parallel_forest_or_root_stat_merge") is not False:
        return False, "root_parallel_forest_or_merge_forbidden"
    if lane.get("shared_frozen_leaf_broker") is not True:
        return False, "shared_frozen_leaf_broker_unproven"
    if lane.get("virtual_loss_or_path_and_leaf_reservations_enabled") is not True:
        return False, "virtual_loss_or_reservations_unproven"
    if lane.get("same_world_inflight_model_evaluation_coalescing_enabled") is not True:
        return False, "same_world_inflight_model_dedup_unproven"
    if lane.get("native_semantic_state_equivalence_required") is not True:
        return False, "native_semantic_dedup_equivalence_unproven"
    if lane.get("public_lookalike_cross_world_merges_prevented") is not True:
        return False, "hidden_or_random_public_lookalike_merge_unproven"
    if lane.get("partial_lane_statistics_used") is not False:
        return False, "partial_lane_statistics_forbidden"
    if lane.get("stock_search_state_isolation_preflight_result") != "passed":
        return False, "stock_search_state_isolation_not_passed"
    tree_identity = lane.get("shared_logical_tree_identity_or_equivalent_integrity_receipt")
    model_identity = lane.get("frozen_model_identity_or_checksum")
    decision_fingerprint = lane.get("decision_fingerprint")
    if not isinstance(tree_identity, str) or not tree_identity:
        return False, "shared_tree_identity_missing"
    if not isinstance(model_identity, str) or not model_identity.startswith("sha256:"):
        return False, "frozen_model_identity_missing"
    if not isinstance(decision_fingerprint, str) or not decision_fingerprint:
        return False, "shared_tree_decision_fingerprint_missing"

    required_counters = (
        "leaf_microbatch_count",
        "lane_trajectory_count",
        "lane_backup_count",
        "virtual_loss_or_path_leaf_reservation_count",
        "in_flight_frozen_eval_coalescing_count",
        "safe_frozen_eval_cache_hit_count",
        "unavoidable_repeat_expansion_count",
        "outstanding_path_or_leaf_reservation_count_at_action_return",
        "outstanding_virtual_loss_count_at_action_return",
        "public_lookalike_cross_world_merge_count",
    )
    counters = {key: _nonnegative_int(lane.get(key)) for key in required_counters}
    if any(value is None for value in counters.values()):
        return False, "shared_tree_counter_telemetry_missing"
    if counters["outstanding_path_or_leaf_reservation_count_at_action_return"] != 0:
        return False, "outstanding_shared_tree_reservation_at_action_return"
    if counters["outstanding_virtual_loss_count_at_action_return"] != 0:
        return False, "outstanding_virtual_loss_at_action_return"
    if counters["public_lookalike_cross_world_merge_count"] != 0:
        return False, "public_lookalike_cross_world_merge_observed"
    batches = lane.get("leaf_microbatch_size_distribution")
    if not isinstance(batches, list) or not batches or any(
        _nonnegative_int(value) in {None, 0} for value in batches
    ):
        return False, "leaf_microbatch_distribution_missing"
    if counters["leaf_microbatch_count"] != len(batches):
        return False, "leaf_microbatch_count_distribution_mismatch"
    if counters["lane_trajectory_count"] < EIGHT_LANE_COUNT:
        return False, "lane_trajectory_count_incomplete"
    if counters["lane_backup_count"] < EIGHT_LANE_COUNT:
        return False, "lane_backup_count_incomplete"
    if not preflight and lane.get("genuine_multistep_mcts") is not True:
        return False, "genuine_multistep_mcts_unproven"
    if not preflight:
        depth = _nonnegative_int(lane.get("max_simulator_search_depth"))
        multistep = _nonnegative_int(lane.get("multi_step_simulations"))
        per_lane = lane.get("per_lane_lifecycle")
        if depth is None or depth < 2 or multistep is None or multistep < 1:
            return False, "non_imputed_multistep_search_evidence_missing"
        if not isinstance(per_lane, list) or len(per_lane) != EIGHT_LANE_COUNT:
            return False, "per_lane_search_lifecycle_missing"
        lane_ids: set[int] = set()
        for lifecycle in per_lane:
            if not isinstance(lifecycle, Mapping):
                return False, "per_lane_search_lifecycle_malformed"
            lane_id = _nonnegative_int(lifecycle.get("lane_id"))
            begin = _nonnegative_int(lifecycle.get("search_begin_calls"))
            release = _nonnegative_int(lifecycle.get("search_release_calls"))
            end = _nonnegative_int(lifecycle.get("search_end_calls"))
            if lane_id is None or begin is None or begin < 1 or release is None or release < 1 or end != 1:
                return False, "per_lane_stock_search_lifecycle_incomplete"
            lane_ids.add(lane_id)
        if lane_ids != set(range(EIGHT_LANE_COUNT)):
            return False, "per_lane_stock_search_ids_incomplete"
    return True, None


def _randomness_receipt_valid(receipt: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Reject omitted or unsafe pre-random-boundary accounting."""

    if receipt.get("complete") is not True:
        return False, "randomness_receipt_incomplete"
    required_zero = (
        "private_random_outcome_samples",
        "guessed_random_rules_or_successors",
        "unobserved_random_outcome_advances",
    )
    for key in required_zero:
        count = _nonnegative_int(receipt.get(key))
        if count is None:
            return False, f"randomness_counter_missing:{key}"
        if count != 0:
            return False, "forbidden_private_randomness_observed"
    for key in (
        "finite_chance_enumerations",
        "unforceable_random_pre_boundary_leaf_evaluations",
    ):
        if _nonnegative_int(receipt.get(key)) is None:
            return False, f"randomness_counter_missing:{key}"
    reasons = receipt.get("unforceable_random_boundary_reasons")
    if not isinstance(reasons, Mapping) or any(
        _nonnegative_int(value) is None for value in reasons.values()
    ):
        return False, "unforceable_random_boundary_reasons_missing"
    if receipt.get("private_unforceable_chance_samples_prohibited") is not True:
        return False, "private_randomness_prohibition_missing"
    if receipt.get("seed_hunting_or_pre_randomization_prohibited") is not True:
        return False, "seed_hunting_prohibition_missing"
    return True, None


def _selected_shared_tree_root_edge_valid(
    row: Mapping[str, Any], lane: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Require action authority from a completed edge in this shared tree.

    A successful eight-lane batch is not sufficient on its own.  The actual
    real-game action has to be one of the current complete legal actions and
    have at least one completed backup on the *same* shared logical tree.
    """

    receipt = row.get("selected_root_action_receipt")
    if not isinstance(receipt, Mapping):
        return False, "selected_shared_tree_root_edge_receipt_missing"
    action = receipt.get("selected_action")
    legal = receipt.get("complete_root_legal_actions")
    if not isinstance(action, list) or not action:
        return False, "selected_shared_tree_action_missing"
    if not isinstance(legal, list) or not legal:
        return False, "selected_shared_tree_legal_actions_missing"
    try:
        canonical_action = tuple(int(value) for value in action)
        canonical_legal = [tuple(int(value) for value in value_row) for value_row in legal]
    except (TypeError, ValueError):
        return False, "selected_shared_tree_action_or_legal_actions_malformed"
    if canonical_action not in canonical_legal:
        return False, "selected_shared_tree_action_not_currently_legal"
    if receipt.get("selected_action_legal") is not True:
        return False, "selected_shared_tree_legal_receipt_missing"
    if receipt.get("selected_action_fully_backed_up") is not True:
        return False, "selected_shared_tree_completed_backup_receipt_missing"
    visits = _nonnegative_int(receipt.get("selected_action_visit_count"))
    backups = _nonnegative_int(receipt.get("selected_action_completed_backups"))
    if visits is None or visits < 1 or backups is None or backups < 1:
        return False, "selected_shared_tree_edge_has_no_completed_backup"
    tree_identity = lane.get("shared_logical_tree_identity_or_equivalent_integrity_receipt")
    if receipt.get("shared_logical_tree_identity") != tree_identity:
        return False, "selected_shared_tree_edge_identity_mismatch"
    if receipt.get("decision_fingerprint") != lane.get("decision_fingerprint"):
        return False, "selected_shared_tree_decision_fingerprint_mismatch"
    try:
        executed = tuple(int(value) for value in row.get("executed_action"))
    except (TypeError, ValueError):
        return False, "executed_mcts_action_missing"
    if executed != canonical_action:
        return False, "executed_mcts_action_differs_from_backed_up_root_edge"
    return True, None


def _preflight_valid(document: Mapping[str, Any], *, portable_smoke: bool) -> tuple[bool, str | None]:
    if document.get("worker_failure") is True:
        return False, str(document.get("failure_reason") or "worker_failure")
    expected_operation = "portable_smoke" if portable_smoke else "preflight"
    if document.get("schema") != PREFLIGHT_SCHEMA or document.get("operation") != expected_operation:
        return False, "preflight_schema_or_operation_mismatch"
    if document.get("status") != "passed":
        return False, "preflight_not_passed"
    stock = document.get("stock_libcg_abi")
    lanes = document.get("eight_lane_capability")
    if not isinstance(stock, Mapping) or not isinstance(lanes, Mapping):
        return False, "missing_stock_or_lane_capability"
    if stock.get("sha256") != STOCK_LIBCG_SHA256 or stock.get("bytes") != STOCK_LIBCG_BYTES:
        return False, "stock_library_identity_mismatch"
    if stock.get("battle_start_seeded_exported") is not False:
        return False, "nonstock_start_export_present"
    lanes_valid, lanes_reason = _shared_tree_lane_receipt_valid(lanes, preflight=True)
    if not lanes_valid:
        return False, lanes_reason
    if portable_smoke and document.get("stock_search_smoke", {}).get("passed") is not True:
        return False, "stock_search_smoke_not_passed"
    return True, None


def _game_valid(document: Mapping[str, Any]) -> tuple[bool, str | None]:
    if document.get("worker_failure") is True:
        return False, str(document.get("failure_reason") or "worker_failure")
    if document.get("schema") != GAME_SCHEMA or document.get("terminal_status") != "completed":
        return False, "game_not_terminal"
    if document.get("invalid_action") is True or document.get("crash") is True:
        return False, "invalid_action_or_crash"
    if document.get("rng_pairing") != "independent_unmatched":
        return False, "rng_pairing_not_truthful"
    game = document.get("game_transport")
    if not isinstance(game, Mapping) or game.get("fresh_one_game_stock_process") is not True:
        return False, "not_fresh_one_game_stock_process"
    stock = document.get("stock_libcg_abi")
    if not isinstance(stock, Mapping) or stock.get("sha256") != STOCK_LIBCG_SHA256:
        return False, "stock_library_receipt_missing"
    if stock.get("battle_start_seeded_exported") is not False:
        return False, "nonstock_start_export_present"
    if document.get("first_player_seat") != document.get("expected_actual_first_seat"):
        return False, "turn_order_wrapper_did_not_enforce_actual_first"
    rows = document.get("experimental_turn_receipts")
    closes = document.get("experimental_actual_turn_close_receipts")
    if not isinstance(rows, list) or not isinstance(closes, list):
        return False, "missing_r222_turn_telemetry"
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
    if not rows:
        return False, "game_has_no_experimental_mcts_decision_rows"
    saw_mcts_action_authority = False
    for row in rows:
        if not isinstance(row, Mapping):
            return False, "malformed_turn_row"
        if row.get("fresh_mcts_search_executed") is True:
            if float(row.get("effective_search_segment_allowance_s", 0.0) or 0.0) > SEARCH_SEGMENT_SECONDS + 1e-6:
                return False, "search_segment_receipt_exceeds_15_seconds"
            lane = row.get("lane_receipt")
            random_receipt = row.get("randomness_receipt")
            if not isinstance(lane, Mapping) or not isinstance(random_receipt, Mapping):
                return False, "missing_lane_or_randomness_receipt"
            lane_valid, lane_reason = _shared_tree_lane_receipt_valid(
                lane, preflight=False
            )
            if not lane_valid:
                return False, lane_reason
            randomness_valid, randomness_reason = _randomness_receipt_valid(
                random_receipt
            )
            if not randomness_valid:
                return False, randomness_reason
            if row.get("mcts_selected_action") is True:
                if row.get("direct_policy_fallback_used") is True:
                    return False, "mcts_authority_and_direct_fallback_both_claimed"
                if row.get("mcts_action_authority_used") is not True:
                    return False, "mcts_selected_action_authority_not_explicit"
                edge_valid, edge_reason = _selected_shared_tree_root_edge_valid(row, lane)
                if not edge_valid:
                    return False, edge_reason
                saw_mcts_action_authority = True
            elif (
                row.get("mcts_selected_action") is False
                and row.get("direct_policy_fallback_used") is True
                and row.get("mcts_action_authority_used") is False
            ):
                pass
            else:
                return False, "fresh_search_row_lacks_explicit_mcts_or_direct_authority_disposition"
    if any(
        float(close.get("effective_actual_turn_planner_pool_seconds", 0.0) or 0.0)
        > TURN_POOL_SECONDS + 1e-6
        for close in closes
        if isinstance(close, Mapping)
    ):
        return False, "actual_turn_pool_receipt_exceeds_45_seconds"
    if not saw_mcts_action_authority:
        return False, "game_has_no_actual_shared_tree_mcts_action_authority"
    return True, None


def _game_record(
    *,
    document: Mapping[str, Any],
    child: Mapping[str, Any],
    pair_index: int,
    game_index: int,
    device: str,
) -> dict[str, Any]:
    valid, reason = _game_valid(document)
    rows = [row for row in document.get("experimental_turn_receipts", []) if isinstance(row, Mapping)]
    closes = [row for row in document.get("experimental_actual_turn_close_receipts", []) if isinstance(row, Mapping)]
    return {
        "pair_index": pair_index,
        "game_index": game_index,
        "mcts_seat": document.get("experimental_seat"),
        "mcts_actual_turn_order": document.get("experimental_actual_turn_order"),
        "winner_seat": document.get("winner_seat"),
        "valid": valid,
        "reason": reason,
        "physical_cuda_device": device,
        "child": dict(child),
        "mcts_turn_closures": closes,
        "mcts_rows": rows,
        "stock_engine_rng_stream": document.get("stock_engine_rng_stream"),
        "mcts_simulations": sum(int(row.get("sims", 0) or 0) for row in rows),
        "mcts_leaf_evaluations": sum(int(row.get("leaf_evaluations", 0) or 0) for row in rows),
        "mcts_root_visits": sum(int(row.get("root_visits", 0) or 0) for row in rows),
        "fallbacks": sum(bool(row.get("direct_policy_fallback_used")) for row in rows),
        "converged_searches": sum(bool(row.get("root_action_stable")) for row in rows),
        "mcts_action_changes_relative_to_frozen_direct_policy": sum(
            bool(row.get("mcts_changed_action_relative_to_frozen_direct_policy")) for row in rows
        ),
        "timing_breaches": sum(bool(row.get("timing_breach_observed")) for row in rows),
        "max_depth": max((int(row.get("max_depth", 0) or 0) for row in rows), default=0),
        "first_search_segments": sum(
            bool(row.get("fresh_mcts_search_executed")) and int(row.get("search_segment_index", 0) or 0) == 1
            for row in rows
        ),
        "later_search_segments": sum(
            bool(row.get("fresh_mcts_search_executed")) and int(row.get("search_segment_index", 0) or 0) > 1
            for row in rows
        ),
    }


def _run_pair(
    *,
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    pair_index: int,
    games_dir: Path,
    logs_dir: Path,
    device: str,
) -> list[dict[str, Any]]:
    schedule = _pair_schedule(EVALUATION_ID, pair_index)
    pair_dir = games_dir / f"pair-{pair_index:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for game in schedule["games"]:
        request = {
            "operation": "run_game",
            "evaluation_id": EVALUATION_ID,
            "pair_nonce_sha256": schedule["pair_nonce_sha256"],
            "direct_package_root": runtime["direct_package_root"],
            "mcts_package_root": runtime["mcts_package_root"],
            "game": game,
        }
        output = pair_dir / f"game-{game['game_index']}.json"
        document, child = _run_child(
            args=args,
            runtime=runtime,
            request=request,
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
                device=device,
            )
        )
    return records


def _prefix_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prefix = [record for record in records if int(record.get("pair_index", -1)) < PREFIX_PAIR_COUNT]
    valid = [record for record in prefix if record.get("valid") is True]
    return {
        "schema": SCHEMA,
        "kind": "live_prefix_diagnostic",
        "pair_indices_inclusive": [0, PREFIX_PAIR_COUNT - 1],
        "completed_games": len(prefix),
        "valid_games": len(valid),
        "mcts_wins": sum(record.get("winner_seat") == record.get("mcts_seat") for record in valid),
        "direct_wins": sum(
            record.get("winner_seat") in {0, 1} and record.get("winner_seat") != record.get("mcts_seat")
            for record in valid
        ),
        "diagnostic_only": True,
        "is_authorization_gate": False,
        "remainder_paused": False,
        "remainder_restarted": False,
        "remainder_reauthorized": False,
        "updated_at_utc": _utc_now(),
    }


def _summary(
    *,
    started: float,
    records: Sequence[Mapping[str, Any]],
    workers: int,
    preflight: Mapping[str, Any] | None,
    portable_smoke: Mapping[str, Any] | None,
) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid") is True]
    rows = [
        row
        for record in valid
        for row in record.get("mcts_rows", [])
        if isinstance(row, Mapping)
    ]
    closures = [
        close
        for record in valid
        for close in record.get("mcts_turn_closures", [])
        if isinstance(close, Mapping)
    ]
    elapsed = max(1e-9, time.monotonic() - started)
    turns = len(closures)
    segment_counts = [int(close.get("search_segments_this_turn", 0) or 0) for close in closures]
    lane_rows = [
        row
        for row in rows
        if row.get("fresh_mcts_search_executed") is True
        and isinstance(row.get("lane_receipt"), Mapping)
    ]
    microbatches = [
        int(size)
        for row in lane_rows
        for size in (row.get("lane_receipt") or {}).get(
            "leaf_microbatch_size_distribution", []
        )
        if isinstance(size, (int, float))
    ]

    def lane_sum(field: str) -> int:
        # Every row in ``lane_rows`` passed _shared_tree_lane_receipt_valid,
        # so this is aggregation of explicit counters, not imputed telemetry.
        return sum(
            int((row.get("lane_receipt") or {})[field]) for row in lane_rows
        )

    microbatch_distribution: dict[str, int] = {}
    for size in microbatches:
        label = str(size)
        microbatch_distribution[label] = microbatch_distribution.get(label, 0) + 1
    prefix = _prefix_summary(records)
    rate = len(records) * 3600.0 / elapsed
    return {
        "schema": SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "mode": "single_bo1000",
        "status": "complete" if len(records) == TOTAL_GAMES else "running",
        "owner_decision_revision": OWNER_DECISION_REVISION,
        "completed_games": len(records),
        "valid_games": len(valid),
        "invalid_games": len(records) - len(valid),
        "total_games": TOTAL_GAMES,
        "seat_swapped_pairs": TOTAL_PAIRS,
        "active_worker_limit": workers,
        "rng_pairing": "independent_unmatched",
        "paired_rng_or_identical_seed_claimed": False,
        "mcts_wins": sum(record.get("winner_seat") == record.get("mcts_seat") for record in valid),
        "direct_wins": sum(
            record.get("winner_seat") in {0, 1} and record.get("winner_seat") != record.get("mcts_seat")
            for record in valid
        ),
        "draws": sum(record.get("winner_seat") not in {0, 1} for record in valid),
        "mcts_as_seat_0_games": sum(record.get("mcts_seat") == 0 for record in valid),
        "mcts_as_seat_1_games": sum(record.get("mcts_seat") == 1 for record in valid),
        "mcts_actual_first_games": sum(record.get("mcts_actual_turn_order") == "first" for record in valid),
        "mcts_actual_second_games": sum(record.get("mcts_actual_turn_order") == "second" for record in valid),
        "total_mcts_turns": turns,
        "turns_with_exactly_one_search_segment": sum(count == 1 for count in segment_counts),
        "turns_with_one_or_more_later_research_segments": sum(
            int(close.get("later_research_count_this_turn", 0) or 0) >= 1 for close in closures
        ),
        "average_search_segments_per_turn": sum(segment_counts) / turns if turns else 0.0,
        "maximum_search_segments_per_turn": max(segment_counts, default=0),
        "first_search_segments": sum(int(record.get("first_search_segments", 0) or 0) for record in records),
        "later_search_segments": sum(int(record.get("later_search_segments", 0) or 0) for record in records),
        "cache_only_later_steps": sum(int(close.get("cache_only_later_steps", 0) or 0) for close in closures),
        "finite_chance_enumerations": sum(int(row.get("finite_chance_enumerations", 0) or 0) for row in rows),
        "unforceable_random_pre_boundary_leaf_evaluations": sum(
            int(row.get("unforceable_random_pre_boundary_leaf_evaluations", 0) or 0) for row in rows
        ),
        "unforceable_random_boundary_reasons": {
            reason: sum(
                int((row.get("unforceable_random_boundary_reasons") or {}).get(reason, 0) or 0)
                for row in rows
            )
            for reason in sorted({
                key for row in rows for key in (row.get("unforceable_random_boundary_reasons") or {})
            })
        },
        "private_random_outcome_samples": sum(int(row.get("private_random_outcome_samples", 0) or 0) for row in rows),
        "guessed_random_rules_or_successors": sum(int(row.get("guessed_random_rules_or_successors", 0) or 0) for row in rows),
        "unobserved_random_outcome_advances": sum(int(row.get("unobserved_random_outcome_advances", 0) or 0) for row in rows),
        "chance_or_information_rebuilds": sum(bool(row.get("chance_or_information_rebuild")) for row in rows),
        "simulations": sum(int(record.get("mcts_simulations", 0) or 0) for record in records),
        "leaf_evaluations": sum(int(record.get("mcts_leaf_evaluations", 0) or 0) for record in records),
        "root_visits": sum(int(record.get("mcts_root_visits", 0) or 0) for record in records),
        "depth": {
            "maximum": max((int(record.get("max_depth", 0) or 0) for record in records), default=0),
            "mean_over_search_rows": (
                sum(int(row.get("max_depth", 0) or 0) for row in rows) / len(rows) if rows else 0.0
            ),
        },
        "convergence": {
            "stable_root_searches": sum(bool(row.get("stable_root_convergence")) for row in rows),
            "fully_backed_selected_actions": sum(bool(row.get("selected_action_fully_backed_up")) for row in rows),
        },
        "lanes": {
            "requested": EIGHT_LANE_COUNT,
            "fresh_mcts_segments_with_complete_eight_lane_receipt": len(lane_rows),
            "active_values": sorted({
                int((row.get("lane_receipt") or {}).get("active_lane_count", 0) or 0)
                for row in lane_rows
            }),
            "shared_logical_tree_segments": sum(
                (row.get("lane_receipt") or {}).get("shared_logical_tree") is True
                for row in lane_rows
            ),
            "root_parallel_forest_or_post_merge_segments": sum(
                (row.get("lane_receipt") or {}).get(
                    "independent_root_parallel_forest_or_root_stat_merge"
                ) is True
                for row in lane_rows
            ),
            "isolated_stock_search_states": lane_sum(
                "isolated_stock_search_state_count"
            ) if lane_rows else 0,
            "lane_trajectories": lane_sum("lane_trajectory_count") if lane_rows else 0,
            "lane_backups": lane_sum("lane_backup_count") if lane_rows else 0,
            "leaf_microbatch_count": len(microbatches),
            "leaf_microbatch_mean_size": sum(microbatches) / len(microbatches) if microbatches else 0.0,
            "leaf_microbatch_max_size": max(microbatches, default=0),
            "leaf_microbatch_size_distribution": microbatch_distribution,
            "virtual_loss_or_path_leaf_reservations": lane_sum(
                "virtual_loss_or_path_leaf_reservation_count"
            ) if lane_rows else 0,
            "in_flight_frozen_eval_coalescing": lane_sum(
                "in_flight_frozen_eval_coalescing_count"
            ) if lane_rows else 0,
            "safe_frozen_eval_cache_hits": lane_sum(
                "safe_frozen_eval_cache_hit_count"
            ) if lane_rows else 0,
            "unavoidable_repeat_expansions": lane_sum(
                "unavoidable_repeat_expansion_count"
            ) if lane_rows else 0,
            "outstanding_path_or_leaf_reservations_at_action_return": sorted({
                int((row.get("lane_receipt") or {}).get(
                    "outstanding_path_or_leaf_reservation_count_at_action_return", 0
                ) or 0)
                for row in lane_rows
            }),
            "outstanding_virtual_loss_at_action_return": sorted({
                int((row.get("lane_receipt") or {}).get(
                    "outstanding_virtual_loss_count_at_action_return", 0
                ) or 0)
                for row in lane_rows
            }),
            "public_lookalike_cross_world_merges": lane_sum(
                "public_lookalike_cross_world_merge_count"
            ) if lane_rows else 0,
            "telemetry_is_aggregate_of_complete_valid_fresh_mcts_rows_only": True,
        },
        "direct_fallbacks": sum(int(record.get("fallbacks", 0) or 0) for record in records),
        "mcts_action_changes_relative_to_frozen_direct_policy": sum(
            int(record.get("mcts_action_changes_relative_to_frozen_direct_policy", 0) or 0)
            for record in records
        ),
        "timing_breaches": sum(int(record.get("timing_breaches", 0) or 0) for record in records),
        "child_timeouts": sum(bool((record.get("child") or {}).get("timed_out")) for record in records),
        "live_prefix_diagnostic": prefix,
        "preflight": dict(preflight or {}),
        "stock_portable_kaggle_runtime_compatibility_smoke": dict(portable_smoke or {}),
        "games_per_hour": rate,
        "eta_seconds": max(0, TOTAL_GAMES - len(records)) * 3600.0 / rate if rate > 0 else None,
        "elapsed_seconds": elapsed,
        "labels": {
            "local_approximate_belief_mcts_non_exact": True,
            "root_sampled_belief_mcts_non_r207_exact_chance": True,
            "non_promotion_exploratory_result": True,
            "frozen_r195_no_rtp_both_arms": True,
            "matchup_adapter_runtime_on_both_arms": True,
            "stock_libcg_only": True,
            "rng_streams_independent_unmatched": True,
            "training_eligible": False,
            "kaggle_submission_authority": False,
        },
        "updated_at_utc": _utc_now(),
    }


def _run_preflight(
    *,
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    output_root: Path,
    device: str,
    portable_smoke: bool,
    required_for_local_bo1000: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool, str | None]:
    operation = "portable_smoke" if portable_smoke else "preflight"
    document, child = _run_child(
        args=args,
        runtime=runtime,
        request={
            "operation": operation,
            "evaluation_id": EVALUATION_ID,
            "direct_package_root": runtime["direct_package_root"],
            "mcts_package_root": runtime["mcts_package_root"],
        },
        result_path=output_root / "preflight" / f"{operation}.json",
        log_path=output_root / "logs" / f"{operation}.log",
        device=device,
    )
    valid, reason = _preflight_valid(document, portable_smoke=portable_smoke)
    if not valid and required_for_local_bo1000:
        raise R222RunnerError(f"{operation} failed closed: {reason}")
    return document, child, valid, reason


def _parent(args: argparse.Namespace) -> int:
    runtime = _validate_source(args)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise R222RunnerError(f"refusing to reuse evaluation output root: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "games").mkdir()
    (output_root / "logs").mkdir()
    _write_json_atomic(
        output_root / "run-contract.json",
        {
            "schema": SCHEMA,
            "evaluation_id": EVALUATION_ID,
            "owner_decision_revision": OWNER_DECISION_REVISION,
            "mode": args.mode,
            "total_games": TOTAL_GAMES,
            "seat_swapped_pairs": TOTAL_PAIRS,
            "workers": args.workers,
            "cuda_visible_devices": args.cuda_visible_devices,
            "child_timeout_seconds": args.child_timeout_seconds,
            "runtime": runtime,
            "timing": {
                "game_seconds": GAME_SECONDS,
                "reserve_seconds": GAME_RESERVE_SECONDS,
                "dynamic_game_allowance_formula": "min(45.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))",
                "default_actual_turn_planner_pool_seconds": TURN_POOL_SECONDS,
                "per_meaningful_search_segment_ceiling_seconds": SEARCH_SEGMENT_SECONDS,
                "lane_count_requested": EIGHT_LANE_COUNT,
                "fixed_simulation_target": None,
                "fixed_depth_target": None,
                "emergency_simulation_safety_ceiling": EMERGENCY_SIMULATION_SAFETY_CEILING,
            },
            "live_prefix_diagnostic": {
                "pair_indices_inclusive": [0, PREFIX_PAIR_COUNT - 1],
                "diagnostic_only": True,
                "authorization_gate": False,
                "pause_or_restart_remainder": False,
            },
            "no_kaggle_submission": True,
            "no_remote_service_or_training_action": True,
            "created_at_utc": _utc_now(),
        },
    )
    device = args.cuda_visible_devices
    preflight, preflight_child, _preflight_validated, _preflight_reason = _run_preflight(
        args=args,
        runtime=runtime,
        output_root=output_root,
        device=device,
        portable_smoke=False,
        required_for_local_bo1000=True,
    )
    portable_smoke, portable_child, portable_smoke_valid, portable_smoke_reason = _run_preflight(
        args=args,
        runtime=runtime,
        output_root=output_root,
        device=device,
        portable_smoke=True,
        # The typed r222 contract requires this receipt but explicitly says it
        # cannot gate, pause, or restart the already-authorized local BO1000.
        required_for_local_bo1000=False,
    )
    _write_json_atomic(
        output_root / "preflight" / "combined.json",
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed",
            "preflight": preflight,
            "preflight_child": preflight_child,
            "portable_smoke": portable_smoke,
            "portable_smoke_child": portable_child,
            "portable_smoke_valid": portable_smoke_valid,
            "portable_smoke_failure_reason": portable_smoke_reason,
            "portable_smoke_nonblocking_for_local_bo1000": True,
        },
    )
    if args.mode == "portable-smoke":
        return 0 if portable_smoke_valid else 2
    if args.mode != "bo1000":
        return 0
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    prefix_written = False
    progress = output_root / "progress.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[list[dict[str, Any]]], int] = {
            executor.submit(
                _run_pair,
                args=args,
                runtime=runtime,
                pair_index=pair_index,
                games_dir=output_root / "games",
                logs_dir=output_root / "logs",
                device=device,
            ): pair_index
            for pair_index in range(TOTAL_PAIRS)
        }
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
                        }
                        for game_index in (0, 1)
                    ]
                records.extend(pair_records)
                for record in pair_records:
                    _append_event(progress, {"schema": SCHEMA, "kind": "game_complete", **record})
                if not prefix_written and all(
                    sum(1 for record in records if record.get("pair_index") == index) == 2
                    for index in range(PREFIX_PAIR_COUNT)
                ):
                    prefix = _prefix_summary(records)
                    _write_json_atomic(output_root / "live-prefix-diagnostic.json", prefix)
                    _append_event(progress, prefix)
                    prefix_written = True
                _write_json_atomic(
                    output_root / "summary.json",
                    _summary(
                        started=started,
                        records=records,
                        workers=args.workers,
                        preflight=preflight,
                        portable_smoke=portable_smoke,
                    ),
                )
    summary = _summary(
        started=started,
        records=records,
        workers=args.workers,
        preflight=preflight,
        portable_smoke=portable_smoke,
    )
    summary["status"] = "complete"
    _write_json_atomic(output_root / "summary.json", summary)
    return 0 if len(records) == TOTAL_GAMES else 2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--mode", choices=("bo1000", "preflight", "portable-smoke"), default="bo1000"
    )
    parser.add_argument("--direct-package", type=Path)
    parser.add_argument("--mcts-package", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=TOTAL_PAIRS)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--cuda-visible-devices", default=BLACKWELL_GPU_UUID)
    parser.add_argument("--child-timeout-seconds", type=float, default=CHILD_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.request is None or args.result is None:
            parser.error("--worker requires --request and --result")
        return args
    if args.output_root is None:
        parser.error("parent requires --output-root")
    if args.workers < 1 or args.workers > 96:
        parser.error("--workers must be in 1..96")
    if args.child_timeout_seconds <= 600.0:
        parser.error("--child-timeout-seconds must be greater than 600")
    if args.cuda_visible_devices != BLACKWELL_GPU_UUID:
        parser.error("--cuda-visible-devices must be the bound Blackwell UUID")
    if args.mode == "bo1000" and (args.pair_start != 0 or args.pair_count != TOTAL_PAIRS):
        parser.error("r222 permits only the one whole --pair-start 0 --pair-count 500 launch")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R222RunnerError as exc:
        print(f"r222 local mirror refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
