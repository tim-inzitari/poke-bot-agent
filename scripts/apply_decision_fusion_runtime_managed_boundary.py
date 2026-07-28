#!/usr/bin/env python3
"""Audit and activate trained all-head fusion at one managed RL boundary.

The watcher leaves the active trainer untouched until the requested immutable
iteration commit exists.  It then uses only the declared systemd lifecycle,
audits the exact learner on Inzi and Elmo, materializes a serving-enabled child,
publishes it into the mutable loop ledger, and resumes the same run.  Any
failure restores the exact committed learner and resumes production.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from scripts.apply_decision_fusion_runtime_at_boundary import (  # noqa: E402
    apply_boundary,
)
from scripts.apply_decision_fusion_warmup_managed_boundary import (  # noqa: E402
    _allocate_artifact_dir,
    _assert_elmo_image_tree_matches,
    _assert_runtime_registry_root,
    _assert_tree_matches,
    _atomic_text,
    _install_stop_override,
    _read,
    _remove_stop_override,
    _run,
    _service_value,
    _sha256,
)
from scripts.build_decision_fusion_activation_validation import (  # noqa: E402
    build as build_validation,
)
from scripts.materialize_decision_fusion_runtime_checkpoint import (  # noqa: E402
    materialize as materialize_runtime,
)


SCHEMA = "poke_bot.causal_decision_fusion_managed_runtime_boundary/v1"
ELMO_IMAGE = "poke-bot-truenas-worker:decision-fusion-v1"


def _pass_marker_supersedes_boundary(
    *, terminal_boundary: bool, marker_exists: bool
) -> bool:
    """Only a continuing-run watcher may be superseded by a gate marker.

    A terminal boundary owns the final checkpoint migration.  The trainer can
    publish its flat-parent gate marker before this watcher gets CPU time; that
    marker is historical evidence for the parent and must never prevent the
    runtime-enabled child from being materialized.
    """

    return bool(marker_exists and not terminal_boundary)


def _runtime_enabled_selector(text: str, runtime_root: Path) -> str:
    """Enable serving fusion in the one canonical selector, preserving all else."""
    root = str(runtime_root)
    runtime_line = f"POKEBOT_SPECIALIST_RUNTIME_ROOT={root}"
    python_line = f"PYTHONPATH={root}"
    rows = text.splitlines()
    if rows.count(runtime_line) != 1 or rows.count(python_line) != 1:
        raise RuntimeError(
            "selector does not point exactly once at the audited fusion runtime"
        )
    required = {
        "POKEBOT_DECISION_FUSION_ENABLED": "1",
        "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED": "1",
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE": (
            "receipt_backed_decision_fusion_runtime_v1"
        ),
    }
    output: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = row.split("=", 1)[0]
        if key not in required:
            output.append(row)
            continue
        if key in seen:
            raise RuntimeError(f"selector contains duplicate fusion key: {key}")
        output.append(f"{key}={required[key]}")
        seen.add(key)
    for key, value in required.items():
        if key not in seen:
            output.append(f"{key}={value}")
    return "\n".join(output) + "\n"


def _publish(path: Path, *, status: str, **values: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            {
                "schema": SCHEMA,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                **values,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _audit_local(
    *,
    runtime_root: Path,
    trained: Path,
    output: Path,
    device: str,
    batch_size: int,
    options_per_state: int,
    warmup: int,
    repeats: int,
) -> None:
    _run(
        [
            sys.executable,
            str(runtime_root / "scripts/audit_decision_fusion_checkpoint.py"),
            "--checkpoint",
            str(trained),
            "--output",
            str(output),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--options-per-state",
            str(options_per_state),
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
        ],
        timeout=300,
    )


def _audit_elmo(
    *,
    runtime_root: Path,
    trained: Path,
    output: Path,
    remote_dir: str,
) -> None:
    _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "192.168.1.143",
            f"mkdir -p {shlex.quote(remote_dir)}",
        ],
        timeout=30,
    )
    _run(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o BatchMode=yes",
            str(trained),
            str(runtime_root / "scripts/audit_decision_fusion_checkpoint.py"),
            f"192.168.1.143:{remote_dir}/",
        ],
        timeout=300,
    )
    checkpoint_name = trained.name
    remote_output = f"{remote_dir}/elmo-parity.json"
    command = (
        "sudo -n docker run --rm "
        f"-v {shlex.quote(remote_dir)}:/probe:rw "
        "-e PYTHONPATH=/workspace --entrypoint python "
        f"{ELMO_IMAGE} /probe/audit_decision_fusion_checkpoint.py "
        f"--checkpoint /probe/{shlex.quote(checkpoint_name)} "
        "--output /probe/elmo-parity.json --device cpu "
        "--batch-size 16 --options-per-state 8 --warmup 5 --repeats 20"
    )
    _run(
        ["ssh", "-o", "BatchMode=yes", "192.168.1.143", command],
        timeout=300,
    )
    _run(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o BatchMode=yes",
            f"192.168.1.143:{remote_output}",
            str(output),
        ],
        timeout=120,
    )


def _assert_warmup_learner(path: Path) -> str:
    digest = checkpoint.checkpoint_digest(path)
    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    state = dict(payload.get("model_state_dict") or {})
    final_weight = state.get("decision_fusion.residual.2.weight")
    if not (
        model_config.get("decision_fusion_enabled") is True
        and model_config.get("decision_fusion_runtime_enabled") is False
        and final_weight is not None
        and int(final_weight.count_nonzero().item()) > 0
    ):
        raise RuntimeError(
            "boundary learner is not a trained serving-disabled fusion checkpoint"
        )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--stop-override", type=Path, required=True)
    parser.add_argument("--maintenance-lock", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bert-stage", type=Path, required=True)
    parser.add_argument("--gate-contract", type=Path)
    parser.add_argument("--specialist-archetype")
    parser.add_argument("--measurement-decks")
    parser.add_argument("--matchup-runtime-tree", type=Path)
    parser.add_argument("--exact-gate-receipt", type=Path)
    parser.add_argument(
        "--remote-worker-endpoints",
        default="192.168.1.143:8765,192.168.1.158:8766",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--terminal-boundary", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    args.run_dir = args.run_dir.expanduser().resolve()
    args.runtime_root = args.runtime_root.expanduser().resolve()
    args.selector = args.selector.expanduser().resolve()
    args.stop_override = args.stop_override.expanduser().resolve()
    args.maintenance_lock = args.maintenance_lock.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.status = args.status.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    if args.gate_contract is not None:
        args.gate_contract = args.gate_contract.expanduser().resolve()
    if args.exact_gate_receipt is not None:
        args.exact_gate_receipt = args.exact_gate_receipt.expanduser().resolve()
    if args.matchup_runtime_tree is not None:
        args.matchup_runtime_tree = args.matchup_runtime_tree.expanduser().resolve()
    if args.terminal_boundary and not (
        args.gate_contract is not None
        and str(args.specialist_archetype or "").strip()
        and str(args.measurement_decks or "").strip()
        and args.exact_gate_receipt is not None
        and args.matchup_runtime_tree is not None
        and args.matchup_runtime_tree.is_file()
    ):
        raise RuntimeError(
            "terminal fusion boundary requires its exact gate contract and deck"
        )
    target_next = int(args.after_iteration) + 1
    loop_path = args.run_dir / "loop_state.json"
    commit_path = (
        args.run_dir / "commits" / f"iter_{args.after_iteration:05d}.json"
    )
    pass_markers = (
        args.run_dir / "SPECIALIST_GATE_PASSED",
        args.run_dir / "SPECIALIST_GATE_PASSED.dudunsparce-splus-v1",
    )

    _assert_runtime_registry_root(args.runtime_root)
    _run(
        [
            sys.executable,
            str(args.runtime_root / "scripts/launch_active_specialist.py"),
            "--check",
        ],
        timeout=60,
    )
    _assert_tree_matches(
        args.runtime_root / "poke_bot",
        str(args.bert_stage),
        "bert.local",
    )
    _assert_elmo_image_tree_matches(args.runtime_root / "poke_bot", ELMO_IMAGE)
    observed = int(_read(loop_path).get("next_iteration", -1))
    if observed > target_next:
        raise RuntimeError(
            f"requested boundary passed: target={target_next} observed={observed}"
        )
    _publish(
        args.status,
        status="validated" if args.validate_only else "waiting_for_boundary",
        after_iteration=args.after_iteration,
        target_next_iteration=target_next,
        observed_next_iteration=observed,
    )
    if args.validate_only:
        return 0

    while True:
        if _pass_marker_supersedes_boundary(
            terminal_boundary=bool(args.terminal_boundary),
            marker_exists=any(path.is_file() for path in pass_markers),
        ):
            _publish(args.status, status="superseded_by_specialist_pass")
            return 0
        state = _read(loop_path)
        completed = int(state.get("last_completed_iteration", -1))
        next_iteration = int(state.get("next_iteration", -1))
        if completed >= args.after_iteration:
            if completed != args.after_iteration or next_iteration != target_next:
                raise RuntimeError("target runtime boundary advanced unexpectedly")
            if not commit_path.is_file() or state != _read(commit_path):
                raise RuntimeError("runtime boundary is not an immutable commit")
            break
        if _service_value(args.unit, "ActiveState") not in {
            "active",
            "activating",
        }:
            raise RuntimeError("trainer stopped before runtime fusion boundary")
        time.sleep(max(0.05, args.poll_seconds))

    last_history = list(state.get("history") or [])
    last_row = last_history[-1] if last_history else {}
    stage_gate = (
        last_row.get("stage_gate") if isinstance(last_row, dict) else None
    )
    if (
        not args.terminal_boundary
        and isinstance(stage_gate, dict)
        and stage_gate.get("passed") is True
    ):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if any(path.is_file() for path in pass_markers):
                _publish(args.status, status="superseded_by_specialist_pass")
                return 0
            time.sleep(0.1)
        raise RuntimeError("passed gate commit did not publish terminal marker")

    trained = Path(str((state.get("learner") or {}).get("path") or "")).resolve()
    trained_digest = _assert_warmup_learner(trained)
    if str((state.get("learner") or {}).get("digest") or "") != trained_digest:
        raise RuntimeError("trained fusion learner digest does not match loop ledger")
    artifact_dir = _allocate_artifact_dir(
        args.artifact_root,
        after_iteration=args.after_iteration,
        parent_digest=trained_digest,
    )
    inzi_parity = artifact_dir / "inzi-cpu-parity.json"
    elmo_parity = artifact_dir / "elmo-cpu-parity.json"
    performance = artifact_dir / "inzi-blackwell-performance.json"
    validation = artifact_dir / "activation-validation.json"
    runtime_checkpoint = artifact_dir / "learner-fusion-runtime.pt"
    materialization = artifact_dir / "runtime-materialization.json"
    activation = artifact_dir / "runtime-boundary-activation.json"
    exact_gate_receipt = (
        args.exact_gate_receipt
        if args.exact_gate_receipt is not None
        else artifact_dir / "runtime-exact-gate.json"
    )
    remote_dir = f"/tmp/pokebot-fusion-audit-{trained_digest[-12:]}"

    loop_changed = False
    selector_changed = False
    selector_before = args.selector.read_text(encoding="utf-8")
    stop_authority_installed = False
    maintenance_lock_installed = False
    error = ""
    try:
        _publish(args.status, status="stopping_at_exact_boundary")
        _atomic_text(
            args.maintenance_lock,
            json.dumps(
                {
                    "schema": "poke_bot.managed_training_maintenance/v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "expires_at_epoch": time.time()
                    + (7200.0 if args.terminal_boundary else 900.0),
                    "owner_pid": os.getpid(),
                    "training_service": args.unit,
                    "authority": SCHEMA,
                    "after_iteration": args.after_iteration,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        maintenance_lock_installed = True
        _install_stop_override(args.stop_override)
        stop_authority_installed = True
        if _service_value(args.unit, "RefuseManualStop") != "no":
            raise RuntimeError("one-shot runtime activation authority was not installed")
        _run(["systemctl", "--user", "stop", args.unit], timeout=90)
        if _service_value(args.unit, "ActiveState") not in {"inactive", "failed"}:
            raise RuntimeError("managed trainer did not stop")
        if _read(loop_path) != _read(commit_path):
            raise RuntimeError("loop state changed after exact runtime-boundary stop")

        _publish(args.status, status="auditing_trained_fusion")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    _audit_local,
                    runtime_root=args.runtime_root,
                    trained=trained,
                    output=inzi_parity,
                    device="cpu",
                    batch_size=16,
                    options_per_state=8,
                    warmup=5,
                    repeats=20,
                ),
                pool.submit(
                    _audit_elmo,
                    runtime_root=args.runtime_root,
                    trained=trained,
                    output=elmo_parity,
                    remote_dir=remote_dir,
                ),
                pool.submit(
                    _audit_local,
                    runtime_root=args.runtime_root,
                    trained=trained,
                    output=performance,
                    device="cuda:0",
                    batch_size=128,
                    options_per_state=16,
                    warmup=20,
                    repeats=100,
                ),
            ]
            for future in futures:
                future.result()

        _publish(args.status, status="building_activation_receipt")
        build_validation(
            checkpoint_path=trained,
            parity_audits=[inzi_parity, elmo_parity],
            performance_audit=performance,
            protocol_path=args.protocol,
            output=validation,
        )
        materialize_runtime(
            trained=trained,
            validation_receipt=validation,
            output=runtime_checkpoint,
            receipt=materialization,
        )
        apply_boundary(
            run_dir=args.run_dir,
            trained=trained,
            runtime_checkpoint=runtime_checkpoint,
            validation_receipt=validation,
            materialization_receipt=materialization,
            activation_receipt=activation,
            expected_last_iteration=args.after_iteration,
            service=args.unit,
        )
        loop_changed = True
        _atomic_text(
            args.selector,
            _runtime_enabled_selector(selector_before, args.runtime_root),
        )
        selector_changed = True

        _remove_stop_override(args.stop_override)
        stop_authority_installed = False
        _run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        if args.terminal_boundary:
            if _service_value(args.unit, "RefuseManualStop") != "yes":
                raise RuntimeError(
                    "terminal runtime activation did not restore stop protection"
                )
            _publish(
                args.status,
                status="evaluating_exact_runtime_checkpoint",
                runtime_checkpoint=str(runtime_checkpoint),
                runtime_checkpoint_sha256=checkpoint.checkpoint_digest(
                    runtime_checkpoint
                ),
                exact_gate_receipt=str(exact_gate_receipt),
            )
            _run(
                [
                    sys.executable,
                    str(
                        args.runtime_root
                        / "scripts"
                        / "run_decision_fusion_runtime_exact_gate.py"
                    ),
                    "--run-dir",
                    str(args.run_dir),
                    "--checkpoint",
                    str(runtime_checkpoint),
                    "--activation-receipt",
                    str(activation),
                    "--iteration",
                    str(args.after_iteration),
                    "--contract",
                    str(args.gate_contract),
                    "--specialist-archetype",
                    str(args.specialist_archetype),
                    "--measurement-decks",
                    str(args.measurement_decks),
                    "--matchup-runtime-tree",
                    str(args.matchup_runtime_tree),
                    "--training-service",
                    str(args.unit),
                    "--remote-worker-endpoints",
                    str(args.remote_worker_endpoints),
                    "--output",
                    str(exact_gate_receipt),
                    "--lock",
                    str(args.maintenance_lock.with_name(
                        "decision-fusion-runtime-exact-gate.lock"
                    )),
                ],
                timeout=3600,
            )
            exact_gate = _read(exact_gate_receipt)
            if not (
                exact_gate.get("schema")
                == "poke_bot.causal_decision_fusion_exact_gate/v1"
                and exact_gate.get("complete") is True
                and (exact_gate.get("checkpoint") or {}).get("digest")
                == checkpoint.checkpoint_digest(runtime_checkpoint)
                and exact_gate.get("premium_gate_complete") is True
                and exact_gate.get("official_gate_complete") is True
            ):
                raise RuntimeError(
                    "terminal runtime child lacks its complete exact gate"
                )
            args.maintenance_lock.unlink(missing_ok=True)
            maintenance_lock_installed = False
            _publish(
                args.status,
                status="complete",
                main_pid=0,
                terminal_boundary=True,
                trained_checkpoint=str(trained),
                trained_checkpoint_sha256=trained_digest,
                runtime_checkpoint=str(runtime_checkpoint),
                runtime_checkpoint_sha256=checkpoint.checkpoint_digest(
                    runtime_checkpoint
                ),
                validation_receipt=str(validation),
                materialization_receipt=str(materialization),
                activation_receipt=str(activation),
                exact_gate_receipt=str(exact_gate_receipt),
                premium_gate_passed=bool(
                    exact_gate.get("premium_gate_passed")
                ),
                official_gate_passed=bool(
                    exact_gate.get("official_gate_passed")
                ),
                both_gates_passed=bool(exact_gate.get("both_gates_passed")),
                stop_protection_restored=True,
            )
            return 0
        # The maintenance lock must cover the restart and stability proof so
        # status-143 recovery cannot race the receipt-backed boundary owner.
        _run(["systemctl", "--user", "start", args.unit], timeout=90)
        pid = int(_service_value(args.unit, "MainPID") or 0)
        if (
            pid <= 0
            or _service_value(args.unit, "ActiveState") != "active"
            or _service_value(args.unit, "RefuseManualStop") != "yes"
        ):
            raise RuntimeError("runtime-fusion trainer did not become protected/active")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (
                int(_service_value(args.unit, "MainPID") or 0) != pid
                or _service_value(args.unit, "ActiveState") != "active"
            ):
                raise RuntimeError("runtime-fusion trainer failed stability check")
            time.sleep(1)
        active = dict(_read(loop_path).get("decision_fusion_activation") or {})
        if not (
            active.get("phase") == "runtime_active"
            and active.get("runtime_enabled") is True
            and active.get("serving_eligible") is True
        ):
            raise RuntimeError("runtime-fusion activation vanished after restart")
        args.maintenance_lock.unlink(missing_ok=True)
        maintenance_lock_installed = False
        _publish(
            args.status,
            status="complete",
            main_pid=pid,
            trained_checkpoint=str(trained),
            trained_checkpoint_sha256=trained_digest,
            runtime_checkpoint=str(runtime_checkpoint),
            runtime_checkpoint_sha256=checkpoint.checkpoint_digest(
                runtime_checkpoint
            ),
            validation_receipt=str(validation),
            materialization_receipt=str(materialization),
            activation_receipt=str(activation),
            stop_protection_restored=True,
        )
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        _publish(args.status, status="recovering_after_error", error=error)
        rollback_errors: list[str] = []
        if selector_changed:
            try:
                _atomic_text(args.selector, selector_before)
            except BaseException as rollback_exc:
                rollback_errors.append(
                    "selector: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        if loop_changed:
            try:
                _atomic_text(
                    loop_path,
                    json.dumps(_read(commit_path), indent=2, sort_keys=True) + "\n",
                )
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"loop: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        if stop_authority_installed:
            _remove_stop_override(args.stop_override)
        _run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        if (
            not args.terminal_boundary
            and _service_value(args.unit, "ActiveState") not in {"active", "activating"}
        ):
            _run(["systemctl", "--user", "start", args.unit], timeout=90, check=False)
        active_state = _service_value(args.unit, "ActiveState")
        main_pid = int(_service_value(args.unit, "MainPID") or 0)
        if maintenance_lock_installed:
            args.maintenance_lock.unlink(missing_ok=True)
            maintenance_lock_installed = False
        _publish(
            args.status,
            status=(
                "rolled_back_to_trained_fusion_warmup"
                if active_state in {"active", "activating"} and main_pid > 0
                else "rollback_incomplete"
            ),
            error=error,
            rollback_errors=rollback_errors,
            active_state=active_state,
            main_pid=main_pid,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
