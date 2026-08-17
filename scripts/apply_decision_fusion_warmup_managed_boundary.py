#!/usr/bin/env python3
"""Deploy zero-safe all-head fusion at an exact managed RL boundary.

The watcher never signals processes directly.  It waits for an immutable
iteration commit, temporarily unlocks only the declared trainer unit, stops it
through systemd, updates the two managed remote-worker deployments, publishes
the zero-safe learner pointer, and resumes the same run.  A terminal specialist
pass wins every race and supersedes this migration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
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

from scripts.apply_decision_fusion_at_boundary import apply_boundary  # noqa: E402
from scripts.materialize_decision_fusion_checkpoint import materialize  # noqa: E402


SCHEMA = "poke_bot.causal_decision_fusion_managed_warmup_boundary/v1"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_text(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


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


def _run(
    argv: list[str],
    *,
    timeout: float = 120.0,
    check: bool = True,
    echo_output: bool = True,
) -> str:
    result = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.stdout and echo_output:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result.stdout.strip()


def _service_value(unit: str, key: str) -> str:
    return _run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        timeout=15,
        echo_output=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_tree_manifest(root: Path) -> str:
    rows = [
        (
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()


_REMOTE_TREE_MANIFEST_CODE = (
    "import hashlib,json,sys;"
    "from pathlib import Path;"
    "r=Path(sys.argv[1]);"
    "x=[(p.relative_to(r).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest())"
    " for p in sorted(r.rglob('*.py')) if '__pycache__' not in p.parts];"
    "print(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest())"
)


def _allocate_artifact_dir(
    artifact_root: Path, *, after_iteration: int, parent_digest: str
) -> Path:
    """Allocate an immutable attempt directory without deleting prior evidence."""

    stem = f"after_iter_{after_iteration:05d}-{parent_digest[-12:]}"
    for attempt in range(1, 10_000):
        suffix = "" if attempt == 1 else f"-attempt-{attempt:04d}"
        candidate = artifact_root / f"{stem}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"no immutable artifact attempt slot remains for {stem}")


def _assert_tree_matches(source: Path, destination: str, host: str) -> None:
    local = _python_tree_manifest(source)
    remote = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "python3 -c "
            f"{shlex.quote(_REMOTE_TREE_MANIFEST_CODE)} "
            f"{shlex.quote(destination)}",
        ],
        timeout=30,
    ).splitlines()[-1].strip()
    if remote != local:
        raise RuntimeError(f"{host} staged Python package digest mismatch")


def _assert_elmo_image_tree_matches(source: Path, image: str) -> None:
    local = _python_tree_manifest(source)
    command = (
        "sudo -n docker run --rm --entrypoint python "
        f"{shlex.quote(image)} -c "
        f"{shlex.quote(_REMOTE_TREE_MANIFEST_CODE)} /workspace/poke_bot"
    )
    remote = _run(
        ["ssh", "-o", "BatchMode=yes", "elmo", command],
        timeout=60,
    ).splitlines()[-1].strip()
    if remote != local:
        raise RuntimeError("Elmo staged Python package digest mismatch")


def _wait_endpoint(runtime_root: Path, endpoint: str, timeout: float = 180.0) -> None:
    host, port_text = endpoint.rsplit(":", 1)
    code = (
        "from poke_bot.remote_jobs import RemoteJobClient;"
        f"c=RemoteJobClient({host!r},{int(port_text)},timeout_s=5,"
        "connect_timeout_s=5,control_timeout_s=8);"
        "c.connect();h=c.health();c.close();"
        "assert h.get('ok') is True,h;print(h)"
    )
    deadline = time.monotonic() + timeout
    error = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=runtime_root,
            env={**os.environ, "PYTHONPATH": str(runtime_root)},
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        if result.returncode == 0:
            print(result.stdout.rstrip(), flush=True)
            return
        error = result.stdout.strip()
        time.sleep(2.0)
    raise RuntimeError(f"endpoint {endpoint} did not recover: {error}")


def _selector_for_root(text: str, old_root: Path, new_root: Path) -> str:
    old = str(old_root)
    new = str(new_root)
    if text.count(old) != 2:
        raise RuntimeError(
            "selector must contain the old root exactly in runtime/PYTHONPATH"
        )
    updated = text.replace(old, new)
    if updated.count(new) != 2 or old in updated:
        raise RuntimeError("selector root replacement is not exact")
    required = {
        "POKEBOT_DECISION_FUSION_ENABLED": "1",
        "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED": "0",
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE": (
            "receipt_backed_decision_fusion_warmup_v1"
        ),
    }
    output: list[str] = []
    seen: set[str] = set()
    for row in updated.splitlines():
        key = row.split("=", 1)[0]
        if key not in required:
            output.append(row)
            continue
        if key not in seen:
            output.append(f"{key}={required[key]}")
            seen.add(key)
    for key, value in required.items():
        if key not in seen:
            output.append(f"{key}={value}")
    return "\n".join(output) + "\n"


def _assert_runtime_registry_root(runtime_root: Path) -> None:
    registry = _read(runtime_root / "ops/specialist_runtime_registry_v1.json")
    declared = Path(str(registry.get("runtime_root") or "")).expanduser()
    if not declared.is_absolute() or declared.resolve() != runtime_root.resolve():
        raise RuntimeError(
            "staged specialist registry delegates outside the declared runtime "
            f"root: declared={declared} expected={runtime_root}"
        )


def _install_stop_override(path: Path) -> None:
    _atomic_text(
        path,
        "[Unit]\n"
        "# One-shot receipt-backed decision-fusion boundary authority.\n"
        "RefuseManualStop=no\n",
    )
    _run(["systemctl", "--user", "daemon-reload"], timeout=30)


def _remove_stop_override(path: Path) -> None:
    path.unlink(missing_ok=True)
    _run(["systemctl", "--user", "daemon-reload"], timeout=30, check=False)


def _deploy_bert(stage: Path, live: Path, label: str) -> None:
    _run(
        [
            "ssh", "-o", "BatchMode=yes", "bert.local",
            "rsync", "-a", "--delete", "--exclude", "__pycache__",
            str(stage) + "/", str(live) + "/",
        ],
        timeout=120,
    )
    uid = _run(
        ["ssh", "-o", "BatchMode=yes", "bert.local", "id", "-u"],
        timeout=15,
    )
    try:
        _run(
            [
                "ssh", "-o", "BatchMode=yes", "bert.local",
                "launchctl", "kickstart", "-k",
                f"gui/{uid}/{label}",
            ],
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        # A launchd child can retain the SSH channel even after kickstart has
        # succeeded.  Endpoint identity is proved by _wait_endpoint after
        # every deploy, so a channel-only timeout is not deployment evidence.
        pass


def _deploy_elmo(
    *,
    project_dir: Path,
    host_compose: Path,
    production_compose: Path,
    fusion_compose: Path,
) -> None:
    command = (
        f"cd {project_dir} && "
        "sudo -n docker compose "
        f"-f {host_compose} -f {production_compose} -f {fusion_compose} "
        "up -d --no-build --force-recreate worker"
    )
    _run(
        ["ssh", "-o", "BatchMode=yes", "elmo", command],
        timeout=180,
    )


def _rollback_elmo(
    *, project_dir: Path, host_compose: Path, production_compose: Path
) -> None:
    command = (
        f"cd {project_dir} && "
        "sudo -n docker compose "
        f"-f {host_compose} -f {production_compose} "
        "up -d --no-build --force-recreate worker"
    )
    _run(
        ["ssh", "-o", "BatchMode=yes", "elmo", command],
        timeout=180,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument(
        "--wait-for-completed-collection-iteration",
        type=int,
        default=None,
        help=(
            "After the immutable boundary exists, preserve this iteration's "
            "receipt-backed collection before stopping, and fail if any train "
            "or evaluation artifact appears first."
        ),
    )
    parser.add_argument("--unit", required=True)
    parser.add_argument("--old-runtime-root", type=Path, required=True)
    parser.add_argument("--new-runtime-root", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--stop-override", type=Path, required=True)
    parser.add_argument("--maintenance-lock", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--bert-stage", type=Path, required=True)
    parser.add_argument("--bert-live", type=Path, required=True)
    parser.add_argument(
        "--bert-label", default="com.pokebot.remote-worker-8766"
    )
    parser.add_argument("--elmo-project-dir", type=Path, required=True)
    parser.add_argument("--elmo-host-compose", type=Path, required=True)
    parser.add_argument("--elmo-production-compose", type=Path, required=True)
    parser.add_argument("--elmo-fusion-compose", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    args.run_dir = args.run_dir.expanduser().resolve()
    args.old_runtime_root = args.old_runtime_root.expanduser().resolve()
    args.new_runtime_root = args.new_runtime_root.expanduser().resolve()
    args.selector = args.selector.expanduser().resolve()
    args.stop_override = args.stop_override.expanduser().resolve()
    args.maintenance_lock = args.maintenance_lock.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.status = args.status.expanduser().resolve()
    target_next = int(args.after_iteration) + 1
    loop_path = args.run_dir / "loop_state.json"
    commit_path = args.run_dir / "commits" / f"iter_{args.after_iteration:05d}.json"
    pass_markers = (
        args.run_dir / "SPECIALIST_GATE_PASSED",
        args.run_dir / "SPECIALIST_GATE_PASSED.dudunsparce-splus-v1",
    )

    selector_before = args.selector.read_text(encoding="utf-8")
    selector_after = _selector_for_root(
        selector_before, args.old_runtime_root, args.new_runtime_root
    )
    if not args.new_runtime_root.is_dir():
        raise RuntimeError("new runtime root is absent")
    _assert_runtime_registry_root(args.new_runtime_root)
    _run(
        [
            sys.executable,
            str(args.new_runtime_root / "scripts/launch_active_specialist.py"),
            "--check",
        ],
        timeout=60,
    )
    _assert_tree_matches(
        args.new_runtime_root / "poke_bot",
        str(args.bert_stage),
        "bert.local",
    )
    elmo_image = _run(
        [
            "ssh", "-o", "BatchMode=yes", "elmo",
            "sudo", "-n", "docker", "image", "inspect",
            "poke-bot-truenas-worker:decision-fusion-v1",
            "--format", "{{.Id}}",
        ],
        timeout=30,
    )
    if not elmo_image.startswith("sha256:"):
        raise RuntimeError("Elmo fusion image is absent")
    _assert_elmo_image_tree_matches(
        args.new_runtime_root / "poke_bot",
        "poke-bot-truenas-worker:decision-fusion-v1",
    )
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
        new_runtime_root=str(args.new_runtime_root),
        elmo_image=elmo_image,
    )
    if args.validate_only:
        return 0

    while True:
        if any(path.is_file() for path in pass_markers):
            _publish(args.status, status="superseded_by_specialist_pass")
            return 0
        state = _read(loop_path)
        completed = int(state.get("last_completed_iteration", -1))
        next_iteration = int(state.get("next_iteration", -1))
        if completed >= args.after_iteration:
            if completed != args.after_iteration or next_iteration != target_next:
                raise RuntimeError("target boundary advanced unexpectedly")
            if not commit_path.is_file() or state != _read(commit_path):
                raise RuntimeError("target boundary is not an exact immutable commit")
            break
        if _service_value(args.unit, "ActiveState") not in {
            "active",
            "activating",
        }:
            raise RuntimeError("trainer stopped before fusion boundary")
        time.sleep(max(0.05, args.poll_seconds))

    # A passed gate can be committed just before its marker is published.
    last_history = list(state.get("history") or [])
    last_row = last_history[-1] if last_history else {}
    stage_gate = (
        last_row.get("stage_gate") if isinstance(last_row, dict) else None
    )
    if isinstance(stage_gate, dict) and stage_gate.get("passed") is True:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if any(path.is_file() for path in pass_markers):
                _publish(args.status, status="superseded_by_specialist_pass")
                return 0
            time.sleep(0.1)
        raise RuntimeError("passed gate commit did not publish terminal marker")

    if args.wait_for_completed_collection_iteration is not None:
        collection_iteration = int(args.wait_for_completed_collection_iteration)
        if collection_iteration != target_next:
            raise RuntimeError(
                "completed-collection recovery must target the boundary's exact "
                "next iteration"
            )
        receipt_path = (
            args.run_dir
            / "collection_receipts"
            / f"iter_{collection_iteration:05d}.json"
        )
        candidate_paths = (
            args.run_dir
            / "checkpoints"
            / f"iter_{collection_iteration:05d}.pt",
            args.run_dir / "eval" / f"iter_{collection_iteration:05d}.json",
            args.run_dir / "metrics" / f"iter_{collection_iteration:05d}.json",
            args.run_dir
            / "research_controls"
            / f"iter_{collection_iteration:05d}.json",
        )
        _publish(
            args.status,
            status="waiting_for_completed_collection",
            after_iteration=args.after_iteration,
            collection_iteration=collection_iteration,
            target_next_iteration=target_next,
        )
        while not receipt_path.is_file():
            if any(path.exists() for path in candidate_paths):
                raise RuntimeError(
                    "training/evaluation artifact appeared before the requested "
                    "completed-collection recovery boundary"
                )
            if _service_value(args.unit, "ActiveState") not in {
                "active",
                "activating",
            }:
                raise RuntimeError(
                    "trainer stopped before the completed collection receipt"
                )
            time.sleep(max(0.05, args.poll_seconds))
        receipt = _read(receipt_path)
        shard = (
            args.run_dir
            / "shards"
            / f"iter_{collection_iteration:05d}.jsonl"
        ).resolve()
        receipt_shard = Path(
            str((receipt.get("shard") or {}).get("path") or "")
        ).resolve()
        if (
            receipt.get("schema") != "poke_bot.completed_collection/v1"
            or int(receipt.get("iteration", -1)) != collection_iteration
            or int(receipt.get("requested_games", -1)) != 8192
            or str(receipt.get("checkpoint_digest") or "")
            != str((state.get("learner") or {}).get("digest") or "")
            or str(receipt.get("design_fingerprint_at_collection") or "")
            != str(state.get("design_fingerprint") or "")
            or receipt_shard != shard
            or not shard.is_file()
            or int((receipt.get("stats") or {}).get("retained_source_games", -1))
            != 8192
            or any(path.exists() for path in candidate_paths)
        ):
            raise RuntimeError(
                "completed collection receipt is not an exact, untrained "
                "8,192-game recovery boundary"
            )

    parent = Path(str((state.get("learner") or {}).get("path") or "")).resolve()
    parent_digest = str((state.get("learner") or {}).get("digest") or "")
    if not parent.is_file() or not parent_digest:
        raise RuntimeError("boundary lacks a concrete learner checkpoint")
    artifact_dir = _allocate_artifact_dir(
        args.artifact_root,
        after_iteration=args.after_iteration,
        parent_digest=parent_digest,
    )
    warmup = artifact_dir / "learner-fusion-warmup.pt"
    materialization_receipt = artifact_dir / "materialization.json"
    activation_receipt = artifact_dir / "boundary-activation.json"

    selector_changed = False
    bert_changed = False
    elmo_changed = False
    loop_changed = False
    maintenance_lock_installed = False
    try:
        _publish(args.status, status="stopping_at_exact_boundary")
        _atomic_text(
            args.maintenance_lock,
            json.dumps(
                {
                    "schema": "poke_bot.managed_training_maintenance/v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "expires_at_epoch": time.time() + 900.0,
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
        if _service_value(args.unit, "RefuseManualStop") != "no":
            raise RuntimeError("one-shot stop authority was not installed")
        _run(["systemctl", "--user", "stop", args.unit], timeout=90)
        if _service_value(args.unit, "ActiveState") not in {"inactive", "failed"}:
            raise RuntimeError("managed trainer did not stop")
        state = _read(loop_path)
        if state != _read(commit_path):
            raise RuntimeError("loop state changed after exact-boundary stop")

        _publish(args.status, status="materializing_zero_safe_learner")
        materialize(
            parent=parent,
            output=warmup,
            receipt=materialization_receipt,
            fusion_width=16,
        )
        apply_boundary(
            run_dir=args.run_dir,
            parent=parent,
            migrated=warmup,
            materialization_receipt=materialization_receipt,
            activation_receipt=activation_receipt,
            expected_last_iteration=args.after_iteration,
            service=args.unit,
        )
        loop_changed = True

        _atomic_text(args.selector, selector_after)
        selector_changed = True
        _publish(args.status, status="deploying_managed_fleet")
        # Mark before the operation so a partial rsync/recreate is rolled back.
        bert_changed = True
        _deploy_bert(args.bert_stage, args.bert_live, args.bert_label)
        elmo_changed = True
        _deploy_elmo(
            project_dir=args.elmo_project_dir,
            host_compose=args.elmo_host_compose,
            production_compose=args.elmo_production_compose,
            fusion_compose=args.elmo_fusion_compose,
        )
        _wait_endpoint(args.new_runtime_root, "bert.local:8766")
        _wait_endpoint(args.new_runtime_root, "elmo:8765")

        _run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        _remove_stop_override(args.stop_override)
        if _service_value(args.unit, "RefuseManualStop") != "yes":
            raise RuntimeError("trainer stop protection was not restored")
        # Keep the managed-maintenance lock through startup and the stability
        # proof.  The gate handler otherwise sees the intentional status-143
        # boundary and can race this controller by restarting the old runtime.
        _run(["systemctl", "--user", "start", args.unit], timeout=90)
        pid = int(_service_value(args.unit, "MainPID") or 0)
        if pid <= 0 or _service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("fusion warmup trainer did not become active")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if (
                int(_service_value(args.unit, "MainPID") or 0) != pid
                or _service_value(args.unit, "ActiveState") != "active"
            ):
                raise RuntimeError("fusion warmup trainer failed stability check")
            time.sleep(1)
        activation = dict(_read(loop_path).get("decision_fusion_activation") or {})
        if not (
            activation.get("phase") == "training_warmup"
            and activation.get("runtime_enabled") is False
        ):
            raise RuntimeError("fusion warmup activation vanished after restart")
        args.maintenance_lock.unlink(missing_ok=True)
        maintenance_lock_installed = False
        _publish(
            args.status,
            status="complete",
            main_pid=pid,
            warmup_checkpoint=str(warmup),
            warmup_checkpoint_sha256=_sha256(warmup),
            materialization_receipt=str(materialization_receipt),
            activation_receipt=str(activation_receipt),
            stop_protection_restored=True,
            bert_managed_worker_updated=True,
            elmo_managed_worker_updated=True,
        )
        return 0
    except BaseException as exc:  # fail-safe rollback at a stopped boundary
        error = f"{type(exc).__name__}: {exc}"
        _publish(args.status, status="recovering_after_error", error=error)
        rollback_errors: list[str] = []
        if selector_changed:
            try:
                _atomic_text(args.selector, selector_before)
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"selector: {type(rollback_exc).__name__}: {rollback_exc}"
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
        # The service is Restart=on-failure. Quiesce its scheduled retry while
        # the selector and remote workers are being restored; otherwise it can
        # consume the start-limit budget against a half-rolled-back fleet.
        _run(
            ["systemctl", "--user", "stop", args.unit],
            timeout=90,
            check=False,
        )
        if bert_changed:
            try:
                _deploy_bert(
                    args.bert_stage.parent.parent / "before" / "poke_bot",
                    args.bert_live,
                    args.bert_label,
                )
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"bert_deploy: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        if elmo_changed:
            try:
                _rollback_elmo(
                    project_dir=args.elmo_project_dir,
                    host_compose=args.elmo_host_compose,
                    production_compose=args.elmo_production_compose,
                )
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"elmo_deploy: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        # Remote workers rotate asynchronously.  The trainer's exact startup
        # contract requires both endpoints, so wait for the restored workers
        # before asking systemd to resume production.
        if bert_changed:
            try:
                _wait_endpoint(args.old_runtime_root, "bert.local:8766")
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"bert_ready: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        if elmo_changed:
            try:
                _wait_endpoint(args.old_runtime_root, "elmo:8765")
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"elmo_ready: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        _remove_stop_override(args.stop_override)
        _run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        if _service_value(args.unit, "ActiveState") not in {"active", "activating"}:
            _run(["systemctl", "--user", "start", args.unit], timeout=90, check=False)
        active_state = _service_value(args.unit, "ActiveState")
        main_pid = int(_service_value(args.unit, "MainPID") or 0)
        if maintenance_lock_installed:
            args.maintenance_lock.unlink(missing_ok=True)
            maintenance_lock_installed = False
        rollback_status = (
            "rolled_back_to_previous_runtime"
            if active_state in {"active", "activating"} and main_pid > 0
            else "rollback_incomplete"
        )
        _publish(
            args.status,
            status=rollback_status,
            error=error,
            rollback_errors=rollback_errors,
            active_state=active_state,
            main_pid=main_pid,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
