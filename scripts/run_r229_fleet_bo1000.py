#!/usr/bin/env python3
"""Durable exactly-once fleet dispatcher for the r229 500-pair mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shlex
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_bot.r229_fleet_mirror_metrics import summarize_games

SCHEMA = "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r229_run/v1"
PAIRS = 500
GAMES = 1000
CHECKPOINT = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
CANONICAL_LIBCG_WHEEL = "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
CANONICAL_NATIVE_LIBRARIES = {
    "linux_x86_64": ("cg/libcg.so", "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7", 1_342_400),
    "linux_aarch64": ("cg/libcg-arm64.so", "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2", 1_296_464),
    "macos_arm64": ("cg/libcg.dylib", "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30", 1_245_544),
    "windows_x86_64": ("cg/cg.dll", "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771", 1_525_248),
}
HOST_LOCAL_EXECUTION_DATA_PLANE = "host_local_sealed_package"
LAN_CONTROL_PLANE_ONLY = "claims_heartbeats_and_json_receipts_only"


class R229FleetError(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def _atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_bytes(encoded)
    os.replace(partial, path)


def _create_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())


def _attempt_number(root: Path, game_id: str) -> int:
    numbers: list[int] = []
    for parent, suffix in ((root / "attempts", ".json"), (root / "logs", ".log")):
        for path in parent.glob(f"{game_id}.attempt-*{suffix}"):
            token = path.name.removesuffix(suffix).rsplit("-", 1)[-1]
            if token.isdigit():
                numbers.append(int(token))
    return max(numbers, default=0)


def schedule() -> list[dict[str, int | str]]:
    return [
        {
            "game_id": f"r229-pair-{pair:04d}-game-{game}",
            "pair_index": pair,
            "game_index": game,
            "mcts_seat": game,
        }
        for pair in range(PAIRS)
        for game in (0, 1)
    ]


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise R229FleetError(f"unreadable JSON receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise R229FleetError(f"receipt is not an object: {path}")
    return payload


def _package_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = config.get("package_manifest_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise R229FleetError("fleet config must bind package_manifest_path")
    path = Path(raw_path)
    manifest = _read(path)
    if (
        manifest.get("schema")
        != "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r253_package/v1"
        or manifest.get("status")
        != "sealed_restarting_serial_mcts_leaf_bounded_evaluation_only"
        or manifest.get("owner_goal_revision") != 253
        or manifest.get("bo_lifecycle_revision") != 233
        or manifest.get("canonical_libcg_revision") != 236
        or manifest.get("superseded_two_lane_topology_revision") != 239
        or manifest.get("owner_handle_scoped_search_id_revision") != 244
        or manifest.get("owner_process_lane_recovery_revision") != 249
        or manifest.get("owner_serial_mcts_revision") != 250
        or manifest.get("owner_internal_leaf_boundary_revision") != 252
        or manifest.get("owner_restarting_serial_rollout_revision") != 253
        or manifest.get("native_simulator_worker_process_count") != 1
        or manifest.get("shared_tree_and_frozen_model_remain_in_parent") is not True
        or manifest.get("native_search_calls_in_parent_worker_threads") is not False
        or manifest.get("concurrent_libcg_search_calls_allowed") is not False
        or manifest.get("complete_serial_retry_count_after_fault") != 1
        or manifest.get("failed_partial_tree_reuse_allowed") is not False
        or manifest.get("consumed_worker_error_clears_in_flight_before_drain") is not True
        or manifest.get("cleanup_error_counts_as_terminal_lane_response") is not True
        or manifest.get("received_step_resolves_in_flight_before_parent_leaf_evaluation") is not True
        or manifest.get("coordinator_post_error_wait_is_bounded") is not True
        or manifest.get("search_seconds_per_attempt") != 8.0
        or manifest.get("exhausted_recovery_direct_fallback_is_degraded") is not True
        or manifest.get("clean_full_game_preflight_max_exhausted_recovery_fallbacks") != 0
        or manifest.get("simulator_lane_count") != 1
        or manifest.get("internal_agent_start_arena_count") != 1
        or manifest.get("minimum_search_begin_call_count_per_searched_decision") != 2
        or manifest.get("search_begin_call_count_equals_completed_rollout_count")
        is not True
        or manifest.get("required_handle_identity_count") != 1
        or manifest.get("required_handle_scoped_search_id_chain_count") != 1
        or manifest.get("required_handle_first_search_id_composite_count") != 1
        or manifest.get("handle_scoped_first_search_id_composite_state_array_field")
        != "handle_scoped_first_search_id_composite_states"
        or manifest.get(
            "handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order"
        )
        != ["lane_id", "handle_identity", "first_search_id"]
        or manifest.get("search_begin_identity_scope")
        != "arena_handle_plus_handle_local_search_id"
        or manifest.get("raw_search_id_global_uniqueness_required") is not False
        or manifest.get("logical_frontier_leaf_count_per_frozen_model_batch") != 1
        or manifest.get("partial_frontier_batches_allowed") is not False
        or manifest.get("serial_one_lane_continuation_required") is not False
        or manifest.get("independent_exact_root_restart_per_rollout_required")
        is not True
        or manifest.get("one_new_leaf_or_value_boundary_maximum_per_rollout")
        is not True
        or manifest.get("rollout_search_id_chain_count_equals_rollout_count")
        is not True
        or manifest.get("bounded_release_and_search_end_per_rollout_required")
        is not True
        or manifest.get("minimum_completed_rollouts_for_mcts_action_authority") != 2
        or manifest.get("maximum_rollouts_per_decision") != 1000
        or manifest.get("one_shared_logical_mcts_tree_required") is not True
        or manifest.get("process_parallel_node_evaluation_included") is not False
        or manifest.get("checkpoint_sha256") != CHECKPOINT
        or manifest.get("complete_ordered_action_ceiling") != 65536
        or manifest.get("internal_ordered_action_expansion_ceiling") != 64
        or manifest.get("every_explicit_chance_context_is_pre_random_value_boundary") is not True
        or manifest.get("explicit_chance_probability_distribution_assumed") is not False
        or manifest.get("deterministic_internal_fanout_over_64_is_value_only_boundary") is not True
        or manifest.get("internal_boundary_representative_action_has_no_tree_authority") is not True
        or manifest.get("internal_boundary_has_action_or_child_authority") is not False
        or manifest.get("internal_value_boundary_telemetry_required") is not True
        or manifest.get("r234_kaggle_broker_or_queue_lifecycle_included") is not False
        or manifest.get("kaggle_search_policy_changes_included") is not False
        or manifest.get("r249_bo_process_lane_boundary_included") is not True
        or manifest.get("r250_serial_process_lane_topology_included") is not True
        or manifest.get("r252_internal_leaf_boundary_included") is not True
        or manifest.get("r253_restarting_serial_rollout_included") is not True
        or manifest.get("continuous_single_trajectory_action_authority_allowed")
        is not False
        or manifest.get("training_eligible") is not False
    ):
        raise R229FleetError("sealed package manifest violates the r229 identity")
    wheel = manifest.get("canonical_libcg_wheel")
    if not isinstance(wheel, Mapping) or wheel.get("sha256") != CANONICAL_LIBCG_WHEEL:
        raise R229FleetError("sealed package manifest lacks the canonical r236 wheel")
    observed_libraries = manifest.get("canonical_native_libraries")
    expected_libraries = {
        platform_name: {"path": path_name, "sha256": digest, "size_bytes": size}
        for platform_name, (path_name, digest, size) in CANONICAL_NATIVE_LIBRARIES.items()
    }
    if observed_libraries != expected_libraries:
        raise R229FleetError("sealed package manifest has a mixed or incomplete libcg set")
    payload_sha = manifest.get("package_payload_tree_sha256")
    if not isinstance(payload_sha, str) or not payload_sha.startswith("sha256:"):
        raise R229FleetError("sealed package manifest lacks its payload digest")
    return {
        "package_manifest_path": str(path.resolve()),
        "package_manifest_sha256": _sha(path),
        "package_payload_tree_sha256": payload_sha,
        "canonical_libcg_revision": 236,
        "mcts_topology_revision": 250,
        "search_id_identity_revision": 244,
        "serial_mcts_revision": 250,
        "internal_leaf_boundary_revision": 252,
        "serial_rollout_revision": 253,
        "simulator_lane_count": 1,
        "canonical_libcg_wheel_sha256": CANONICAL_LIBCG_WHEEL,
        "canonical_native_libraries": expected_libraries,
    }


def _complete(path: Path, job: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    row = _read(path)
    return (
        row.get("status") == "complete"
        and row.get("game_id") == job["game_id"]
        and row.get("pair_index") == job["pair_index"]
        and row.get("game_index") == job["game_index"]
        and row.get("mcts_seat") == job["mcts_seat"]
        and row.get("canonical_libcg_revision") == 236
        and row.get("mcts_topology_revision") == 250
        and row.get("search_id_identity_revision") == 244
        and row.get("serial_mcts_revision") == 250
        and row.get("internal_leaf_boundary_revision") == 252
        and row.get("serial_rollout_revision") == 253
        and row.get("simulator_lane_count") == 1
        and row.get("training_eligible") is False
    )


def _host_admitted(host: Mapping[str, Any]) -> tuple[bool, str]:
    command = host.get("admission_command")
    if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
        return False, "missing_admission_command"
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20, check=False)
    expected = str(host.get("admission_exact_stdout", "ADMITTED"))
    if result.returncode != 0 or result.stdout.strip() != expected:
        return False, f"admission_refused:{result.returncode}:{result.stdout.strip()[:200]}"
    return True, "admitted"


def _validate_host_local_execution(
    config: Mapping[str, Any], hosts: Sequence[Mapping[str, Any]]
) -> None:
    """Fail closed unless games consume only pre-staged host-local payloads."""

    if config.get("execution_data_plane") != HOST_LOCAL_EXECUTION_DATA_PLANE:
        raise R229FleetError("fleet execution data plane must be host-local")
    if config.get("lan_control_plane") != LAN_CONTROL_PLANE_ONLY:
        raise R229FleetError("LAN must carry only claims, heartbeats, and JSON receipts")
    if config.get("per_game_package_or_model_transfer_allowed") is not False:
        raise R229FleetError("per-game package or model transfer must be disabled")
    for host in hosts:
        root = host.get("host_local_evaluation_root")
        package = host.get("host_local_package_root")
        command = host.get("command")
        if (
            not isinstance(root, str)
            or not Path(root).is_absolute()
            or not isinstance(package, str)
            or not Path(package).is_absolute()
            or not isinstance(command, list)
            or any(not isinstance(item, str) for item in command)
        ):
            raise R229FleetError(
                f"{host.get('id')} lacks an absolute host-local package binding"
            )
        if host.get("per_game_package_or_model_transfer_allowed") is not False:
            raise R229FleetError(
                f"{host.get('id')} permits per-game package or model transfer"
            )
        if any(item in {"scp", "rsync", "sftp", "curl", "wget"} for item in command):
            raise R229FleetError(
                f"{host.get('id')} command contains a payload-transfer program"
            )
        command_text = "\n".join(command)
        container_package = host.get("container_local_package_root")
        package_is_bound = root in command_text or package in command_text
        if isinstance(container_package, str):
            package_is_bound = package_is_bound and container_package in command_text
        if not package_is_bound:
            raise R229FleetError(
                f"{host.get('id')} command does not bind its host-local package"
            )


def _run(host: Mapping[str, Any], job: Mapping[str, Any], output: Path, log: Path) -> dict[str, Any]:
    admitted, reason = _host_admitted(host)
    if not admitted:
        return {"disposition": "not_admitted", "reason": reason}
    template = host.get("command")
    if not isinstance(template, list) or any(not isinstance(item, str) for item in template):
        raise R229FleetError("host command must be an argv template")
    values = {
        "pair_index": str(job["pair_index"]), "game_index": str(job["game_index"]),
        "mcts_seat": str(job["mcts_seat"]), "game_id": str(job["game_id"]),
        "output": str(output), "host": str(host["id"]),
    }
    argv = [item.format_map(values) for item in template]
    cleanup_template = host.get("failed_child_cleanup_command")

    def cleanup_failed_child() -> str:
        if cleanup_template is None:
            return "not_configured"
        if not isinstance(cleanup_template, list) or any(
            not isinstance(item, str) for item in cleanup_template
        ):
            return "invalid_cleanup_command"
        cleanup_argv = [item.format_map(values) for item in cleanup_template]
        try:
            cleanup = subprocess.run(
                cleanup_argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=30, check=False,
            )
        except Exception as cleanup_exc:  # noqa: BLE001 - preserve evidence
            return f"cleanup_exception:{type(cleanup_exc).__name__}:{cleanup_exc}"
        return f"cleanup_exit:{cleanup.returncode}:{cleanup.stdout.strip()[:200]}"
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        captured, _ = process.communicate(
            timeout=float(host.get("game_timeout_seconds", 3600))
        )
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            captured, _ = process.communicate(
                timeout=float(host.get("timeout_grace_seconds", 20))
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            captured, _ = process.communicate()
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        cleanup_evidence = cleanup_failed_child()
        log.write_text(captured + f"\nR229_FAILED_CHILD_CLEANUP {cleanup_evidence}\n")
        raise R229FleetError(
            f"{host['id']} game {job['game_id']} timed out; {cleanup_evidence}"
        ) from exc
    log.write_text(captured)
    if process.returncode != 0:
        cleanup_evidence = cleanup_failed_child()
        with log.open("a") as stream:
            stream.write(f"\nR229_FAILED_CHILD_CLEANUP {cleanup_evidence}\n")
        raise R229FleetError(
            f"{host['id']} game {job['game_id']} exited {process.returncode}; "
            f"{cleanup_evidence}"
        )
    if not output.is_file():
        receipt = None
        for line in reversed(captured.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("schema") == "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r229_game/v1":
                receipt = candidate
                break
        if receipt is None:
            raise R229FleetError(f"{host['id']} emitted no parseable game receipt for {job['game_id']}")
        _atomic(output, receipt)
    if not _complete(output, job):
        raise R229FleetError(f"{host['id']} returned no exact complete receipt for {job['game_id']}")
    return {"disposition": "complete", "host": host["id"], "wall_seconds": time.monotonic() - started}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.monotonic()
    config = _read(args.config)
    package_identity = _package_identity(config)
    hosts = config.get("hosts")
    if not isinstance(hosts, list):
        raise R229FleetError("fleet config must contain a host list")
    _validate_host_local_execution(config, hosts)
    roles = {row.get("role") for row in hosts if isinstance(row, Mapping)}
    ids = [row.get("id") for row in hosts if isinstance(row, Mapping)]
    if roles != {"elmo", "bert", "train_inzi"} or len(ids) != len(set(ids)):
        raise R229FleetError("fleet config must cover Elmo, Bert, and Train/Inzi with unique slot identities")
    root = args.output_root.resolve()
    games_dir, logs_dir = root / "games", root / "logs"
    jobs = [job for job in schedule() if not _complete(games_dir / f"{job['game_id']}.json", job)]
    events = root / "events.jsonl"
    run_identity = {
        "schema": SCHEMA, "created_at_utc": _utc(), "config_sha256": _sha(args.config),
        "runner_sha256": _sha(Path(__file__).resolve()), "total_pairs": PAIRS,
        "total_games": GAMES, "training_eligible": False, **package_identity,
    }
    identity_path = root / "run-identity.json"
    if identity_path.exists() and _read(identity_path) != run_identity:
        # created_at is intentionally fixed by the first invocation.
        old = _read(identity_path)
        for key in (
            "config_sha256", "runner_sha256", "total_pairs", "total_games",
            "package_manifest_sha256", "package_payload_tree_sha256",
            "canonical_libcg_revision", "canonical_libcg_wheel_sha256",
            "canonical_native_libraries",
            "mcts_topology_revision", "search_id_identity_revision",
            "serial_mcts_revision", "internal_leaf_boundary_revision",
            "serial_rollout_revision",
            "simulator_lane_count",
        ):
            if old.get(key) != run_identity.get(key):
                raise R229FleetError(f"resume identity drifted: {key}")
        run_identity = old
    else:
        _atomic(identity_path, run_identity)

    slots: list[Mapping[str, Any]] = []
    for host in hosts:
        slots.extend([host] * int(host.get("slots", 0)))
    if not slots:
        raise R229FleetError("fleet has no configured slots")
    pending = list(jobs)
    futures: dict[
        Future[dict[str, Any]],
        tuple[Mapping[str, Any], Mapping[str, Any], Path, Path, str, float, int],
    ] = {}
    consecutive_failures = {str(host["id"]): 0 for host in hosts}
    quarantined: set[str] = set()
    attempt_counter: dict[str, int] = {
        str(job["game_id"]): _attempt_number(root, str(job["game_id"]))
        for job in schedule()
    }
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        free = list(slots)
        while pending or futures:
            free = [host for host in free if str(host["id"]) not in quarantined]
            while pending and free:
                host = free.pop(0)
                job = pending.pop(0)
                output = games_dir / f"{job['game_id']}.json"
                attempt_counter[str(job["game_id"])] = attempt_counter.get(str(job["game_id"]), 0) + 1
                attempt = attempt_counter[str(job["game_id"])]
                attempt_path = root / "attempts" / f"{job['game_id']}.attempt-{attempt:03d}.json"
                log_path = logs_dir / f"{job['game_id']}.attempt-{attempt:03d}.log"
                attempt_started_utc = _utc()
                attempt_started = time.monotonic()
                future = pool.submit(_run, host, job, output, log_path)
                futures[future] = (
                    host, job, attempt_path, log_path, attempt_started_utc,
                    attempt_started, attempt,
                )
            if not futures:
                raise R229FleetError("every host refused admission while games remain")
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                (
                    host, job, attempt_path, log_path, attempt_started_utc,
                    attempt_started, attempt,
                ) = futures.pop(future)
                host_id = str(host["id"])
                try:
                    result = future.result()
                except Exception as exc:
                    consecutive_failures[host_id] += 1
                    if consecutive_failures[host_id] >= int(args.quarantine_after_failures):
                        quarantined.add(host_id)
                    result = {
                        "disposition": "failed_attempt_requeued",
                        "error": f"{type(exc).__name__}: {exc}",
                        "consecutive_host_failures": consecutive_failures[host_id],
                        "host_quarantined": host_id in quarantined,
                    }
                    pending.append(job)
                else:
                    if result["disposition"] == "complete":
                        consecutive_failures[host_id] = 0
                    elif result["disposition"] == "not_admitted":
                        pending.append(job)
                _create_once(attempt_path, {
                    "schema": "poke_bot.r229_fleet_game_attempt/v1",
                    "attempt": attempt,
                    "started_at_utc": attempt_started_utc,
                    "completed_at_utc": _utc(),
                    "attempt_wall_seconds": time.monotonic() - attempt_started,
                    "host": host_id,
                    "game": dict(job),
                    "log_path": str(log_path),
                    "log_sha256": _sha(log_path) if log_path.is_file() else None,
                    **result,
                })
                if host_id not in quarantined:
                    free.append(host)
                event = {"at_utc": _utc(), "host": host["id"], "game_id": job["game_id"], **result}
                events.parent.mkdir(parents=True, exist_ok=True)
                with events.open("a") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
                if result["disposition"] == "not_admitted":
                    # Avoid a hot loop if capacity is temporarily protected.
                    time.sleep(float(args.admission_retry_seconds))
    rows = [_read(games_dir / f"{job['game_id']}.json") for job in schedule()]
    summary = summarize_games(rows, require_complete=True)
    fleet_wall = max(1e-9, time.monotonic() - run_started)
    summary["throughput"]["fleet_wall_seconds_this_invocation"] = fleet_wall
    summary["throughput"]["fleet_games_per_second_this_invocation"] = len(jobs) / fleet_wall
    summary["throughput"]["fleet_games_per_hour_this_invocation"] = len(jobs) * 3600.0 / fleet_wall
    final = {**run_identity, "status": "complete", "completed_at_utc": _utc(), "summary": summary}
    _atomic(root / "final-review.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--admission-retry-seconds", type=float, default=30.0)
    parser.add_argument("--quarantine-after-failures", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
