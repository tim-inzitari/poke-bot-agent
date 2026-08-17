#!/usr/bin/env python3
"""Evaluate the exact serving-enabled fusion child at a terminal boundary.

This is deliberately a post-activation transaction.  It does not rewrite the
immutable RL iteration commit or reuse the flat-parent result.  The 2,000-game
premium gate and 1,000-game official gate are rerun against one exact runtime
checkpoint, and an append-only receipt binds both results to the activation
receipt and checkpoint checksum.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "poke_bot.causal_decision_fusion_exact_gate/v1"
ACTIVATION_SCHEMA = "poke_bot.causal_decision_fusion_runtime_boundary/v1"
FORMAL_GATE_SEED_OFFSET = 19_000_000
RESEARCH_CONTROL_SEED_OFFSET = 39_000_000
ITERATION_SEED_STRIDE = 100_000


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _service_stopped(service: str) -> None:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "MainPID",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    values = dict(
        row.split("=", 1)
        for row in completed.stdout.splitlines()
        if "=" in row
    )
    if (
        completed.returncode
        or values.get("ActiveState") not in {"inactive", "failed"}
        or int(values.get("MainPID") or 0) != 0
    ):
        raise RuntimeError(
            f"runtime exact gate requires stopped managed trainer: {values}"
        )


def _activation_identity(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    activation_path: Path,
    iteration: int,
) -> tuple[str, dict[str, Any]]:
    from poke_bot import checkpoint

    digest = checkpoint.checkpoint_digest(checkpoint_path)
    state = _read(run_dir / "loop_state.json")
    activation = _read(activation_path)
    learner = dict(state.get("learner") or {})
    active = dict(state.get("decision_fusion_activation") or {})
    payload = checkpoint.load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    if not (
        activation.get("schema") == ACTIVATION_SCHEMA
        and int(
            (activation.get("boundary") or {}).get(
                "last_completed_iteration", -1
            )
        )
        == int(iteration)
        and str(
            (activation.get("runtime_learner") or {}).get("digest") or ""
        )
        == digest
        and Path(
            str((activation.get("runtime_learner") or {}).get("path") or "")
        ).expanduser().resolve()
        == checkpoint_path
        and learner
        == {"path": str(checkpoint_path), "digest": digest}
        and active.get("phase") == "runtime_active"
        and active.get("runtime_enabled") is True
        and active.get("serving_eligible") is True
        and str(active.get("learner_digest") or "") == digest
        and str(active.get("receipt_digest") or "") == _sha256(activation_path)
        and model_config.get("decision_fusion_enabled") is True
        and model_config.get("decision_fusion_runtime_enabled") is True
    ):
        raise RuntimeError(
            "runtime exact gate checkpoint is not the activated fused learner"
        )
    return digest, activation


def _validate_result(
    *,
    result: dict[str, Any],
    contract: dict[str, Any],
    checkpoint_digest: str,
) -> tuple[bool, bool]:
    gate = dict(contract.get("next_gate") or {})
    evaluation = dict(gate.get("evaluation") or {})
    premium_audit = dict(result.get("audit") or {})
    official = dict(result.get("research_controls") or {})
    official_audit = dict(official.get("audit") or {})
    research_checks = dict(result.get("research_checks") or {})
    premium_ids = {
        str(row.get("opponent_id") or "") for row in gate.get("roster") or []
    }
    official_ids = {
        str(row.get("opponent_id") or "")
        for row in gate.get("research_measurements") or []
    }
    premium_rows = {
        str(row.get("opponent_id") or ""): dict(row)
        for row in result.get("matchups") or []
    }
    official_rows = {
        str(row.get("opponent_id") or ""): dict(row)
        for row in official.get("matchups") or []
    }
    premium_games = int(evaluation.get("games_total") or 0)
    premium_per = int(evaluation.get("games_per_opponent") or 0)
    official_games = sum(
        int(row.get("games") or 0)
        for row in gate.get("research_measurements") or []
    )
    premium_complete = bool(
        result.get("checkpoint_digest") == checkpoint_digest
        and int(result.get("games") or 0) == premium_games
        and premium_audit.get("passed") is True
        and premium_audit.get("exact_distribution") is True
        and premium_audit.get("exact_weights") is True
        and set(premium_rows) == premium_ids
        and all(
            int(row.get("games") or 0) == premium_per
            and int(row.get("seat0") or 0) == premium_per // 2
            and int(row.get("seat1") or 0) == premium_per // 2
            for row in premium_rows.values()
        )
    )
    official_complete = bool(
        int(official.get("games") or 0) == official_games == 1000
        and official_audit.get("passed") is True
        and official_audit.get("exact_distribution") is True
        and official_audit.get("exact_weights") is True
        and official_audit.get("greedy_required") is True
        and set(official_rows) == official_ids
        and all(
            int(row.get("games") or 0) == 250
            and int(row.get("seat0") or 0) == 125
            and int(row.get("seat1") or 0) == 125
            for row in official_rows.values()
        )
    )
    if not premium_complete or not official_complete:
        raise RuntimeError("runtime fused checkpoint lacks two complete exact gates")
    premium_passed = bool(result.get("passed") is True)
    official_passed = bool(
        research_checks.get("research_control_audit") is True
        and research_checks.get("accepted_official_holdout_non_regression")
        is True
    )
    return premium_passed, official_passed


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from poke_bot.baselines_runtime import (
        baseline_content_digest,
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.pure_rl.hardware import full_hardware_profile
    from poke_bot.pure_rl.multi_env_self_play import pure_rl_leaf_coalesce_ms
    from poke_bot.pure_rl.strong_public_gate import (
        build_strong_public_gate_result,
        load_active_gate_contract,
        verify_roster_content,
    )
    from poke_bot.remote_jobs import RemoteWorkerFarm
    from poke_bot.remote_sim_jobs import remote_play_job, remote_self_play_job
    from scripts.train_pure_rl import (
        _LeafFarm,
        _hard_gate_publish_weights,
        _heldout_eval,
        _our_decks,
        _remote_heldout_capability_audit,
        _select_measurement_decks,
    )

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    activation_path = args.activation_receipt.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _service_stopped(args.training_service)
    digest, activation = _activation_identity(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        activation_path=activation_path,
        iteration=int(args.iteration),
    )
    if output.is_file():
        existing = _read(output)
        if (
            existing.get("schema") == SCHEMA
            and (existing.get("checkpoint") or {}).get("digest") == digest
            and existing.get("complete") is True
        ):
            return existing
        raise RuntimeError("existing runtime exact-gate receipt conflicts")

    contract = load_active_gate_contract(contract_path)
    gate = dict(contract["next_gate"])
    ensure_baselines_installed()
    loadable, failed = filter_loadable_baselines(load_manifest())
    if failed:
        print(
            f"[fusion_exact_gate] ignored unrelated unloadable={len(failed)}",
            flush=True,
        )
    by_id = {spec.id: spec for spec in loadable}
    premium_ids = tuple(str(row["opponent_id"]) for row in gate["roster"])
    official_ids = tuple(
        str(row["opponent_id"]) for row in gate["research_measurements"]
    )
    missing = [key for key in (*premium_ids, *official_ids) if key not in by_id]
    if missing:
        raise RuntimeError(f"runtime gate packages unavailable: {missing}")
    premium_specs = [by_id[key] for key in premium_ids]
    official_specs = [by_id[key] for key in official_ids]
    verify_roster_content(
        gate,
        {
            spec.id: baseline_content_digest(spec.path)
            for spec in premium_specs
        },
    )
    decks = _our_decks("specialist", args.specialist_archetype)
    measurement = _select_measurement_decks(decks, args.measurement_decks)
    if [name for name, _deck in measurement] != [args.specialist_archetype]:
        raise RuntimeError("runtime exact gate must use only the active deck")

    hw = full_hardware_profile()
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw.validate_or_raise(visible_gpu_count=visible)
    workers = int(args.workers) if int(args.workers) > 0 else int(hw.sim_workers)
    endpoints = [
        value.strip()
        for value in str(args.remote_worker_endpoints).split(",")
        if value.strip()
    ]
    leaf = _LeafFarm()
    remote_farm = None
    started = time.time()
    try:
        leaf.start(
            ckpt=checkpoint_path,
            digest=digest,
            leaf_devices=hw.leaf_cuda_devices(),
            n_workers=workers,
            max_batch=None,
            coalesce_ms=pure_rl_leaf_coalesce_ms(default=0.0),
        )
        connected: list[str] = []
        if endpoints:
            timeout = max(120.0, float(args.game_timeout_s) + 90.0)
            remote_farm = RemoteWorkerFarm(endpoints, timeout_s=timeout)
            infos = remote_farm.connect(require_all=False)
            connected = [str(info.endpoint) for info in infos]
            if remote_farm.clients:
                capability = _remote_heldout_capability_audit(
                    remote_farm, required_endpoints=connected
                )
                if capability.get("passed") is not True:
                    raise RuntimeError(
                        "formal-eval remote capability audit failed: "
                        + json.dumps(capability, sort_keys=True)
                    )
            else:
                remote_farm.close()
                remote_farm = None
        publish = _hard_gate_publish_weights(
            leaf=leaf,
            remote_farm=remote_farm,
            ckpt=checkpoint_path,
            digest=digest,
            version=int(time.time()),
            required_endpoints=connected,
            reload_local=False,
        )
        premium_seed = (
            FORMAL_GATE_SEED_OFFSET
            + int(args.iteration) * ITERATION_SEED_STRIDE
        )
        official_seed = (
            RESEARCH_CONTROL_SEED_OFFSET
            + int(args.iteration) * ITERATION_SEED_STRIDE
        )
        premium_rows, premium_audit = _heldout_eval(
            ckpt=checkpoint_path,
            digest=digest,
            n_games=int(gate["evaluation"]["games_total"]),
            decks=measurement,
            official_specs=premium_specs,
            seed=premium_seed,
            game_timeout_s=int(args.game_timeout_s),
            n_workers=workers,
            leaf_channel=leaf.remote_channel,
            remote_farm=remote_farm,
            worker_play=remote_play_job,
            worker_self_play=remote_self_play_job,
            mode="specialist",
            allow_remote_play=remote_farm is not None,
            iteration=int(args.iteration),
            gate_wr=float(
                gate["pass_criteria"]["skill_weighted_win_rate"]
            ),
            opponent_ids=premium_ids,
            stage_label="heldout:fused_runtime_premium_gate",
        )
        official_rows, official_audit = _heldout_eval(
            ckpt=checkpoint_path,
            digest=digest,
            n_games=sum(
                int(row["games"]) for row in gate["research_measurements"]
            ),
            decks=measurement,
            official_specs=official_specs,
            seed=official_seed,
            game_timeout_s=int(args.game_timeout_s),
            n_workers=workers,
            leaf_channel=leaf.remote_channel,
            remote_farm=remote_farm,
            worker_play=remote_play_job,
            worker_self_play=remote_self_play_job,
            mode="specialist",
            allow_remote_play=remote_farm is not None,
            iteration=int(args.iteration),
            gate_wr=float(
                gate["pass_criteria"][
                    "accepted_official_holdout_non_regression"
                ]
            ),
            opponent_ids=official_ids,
            stage_label="heldout:fused_runtime_official_gate",
        )
        result = build_strong_public_gate_result(
            contract=contract,
            checkpoint=str(checkpoint_path),
            checkpoint_digest=digest,
            iteration=int(args.iteration),
            gate_rows=premium_rows,
            gate_audit=premium_audit,
            research_rows=official_rows,
            research_audit=official_audit,
            gate_seed=premium_seed,
            research_seed=official_seed,
            bootstrap_resamples=int(args.bootstrap_resamples),
        )
        premium_passed, official_passed = _validate_result(
            result=result,
            contract=contract,
            checkpoint_digest=digest,
        )
        receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "training_eligible": False,
            "replay_eligible": False,
            "run_dir": str(run_dir),
            "iteration": int(args.iteration),
            "boundary": {
                "commit": str(
                    run_dir
                    / "commits"
                    / f"iter_{int(args.iteration):05d}.json"
                ),
                "commit_digest": str(
                    (activation.get("boundary") or {}).get("commit_digest") or ""
                ),
            },
            "activation_receipt": {
                "path": str(activation_path),
                "digest": _sha256(activation_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "digest": digest,
            },
            "contract": {
                "path": str(contract_path),
                "digest": _sha256(contract_path),
                "canonical_digest": _canonical_digest(contract),
                "gate_id": str(gate["id"]),
            },
            "checkpoint_publish_proof": publish,
            "premium_gate_complete": True,
            "official_gate_complete": True,
            "premium_gate_passed": premium_passed,
            "official_gate_passed": official_passed,
            "both_gates_passed": bool(premium_passed and official_passed),
            "completion_authority": (
                "measured_both_gates_pass"
                if premium_passed and official_passed
                else "explicit_owner_ceiling_acceptance"
            ),
            "result": result,
            "result_digest": _canonical_digest(result),
            "elapsed_sec": time.time() - started,
        }
        _exclusive_json(output, receipt)
        print(
            "[fusion_exact_gate] COMPLETE "
            f"checkpoint={digest[:19]}... "
            f"premium_passed={premium_passed} "
            f"official_passed={official_passed} "
            f"receipt={output}",
            flush=True,
        )
        return receipt
    finally:
        leaf.stop()
        if remote_farm is not None:
            try:
                remote_farm.close()
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--specialist-archetype", required=True)
    parser.add_argument("--measurement-decks", required=True)
    parser.add_argument("--matchup-runtime-tree", type=Path, required=True)
    parser.add_argument("--training-service", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--game-timeout-s", type=int, default=600)
    parser.add_argument(
        "--remote-worker-endpoints",
        default="elmo:8765,bert:8766",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    stream = args.lock.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another runtime exact gate owns the lock") from exc
    os.environ["POKEBOT_PRIMARY_ARCHETYPE"] = args.specialist_archetype
    os.environ["POKEBOT_DECISION_FUSION_ENABLED"] = "1"
    os.environ["POKEBOT_DECISION_FUSION_RUNTIME_ENABLED"] = "1"
    matchup_tree = args.matchup_runtime_tree.expanduser().resolve()
    if not matchup_tree.is_file():
        raise FileNotFoundError(matchup_tree)
    os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(matchup_tree)
    os.environ["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] = "runtime"
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    os.environ["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    os.environ.setdefault(
        "PURE_RL_PROGRESS_LOG",
        "/home/pokebot/poke-bot-agent/outputs/logs/"
        "dudunsparce_decision_fusion_runtime_exact_gate.progress.log",
    )
    try:
        run(args)
        return 0
    finally:
        fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
