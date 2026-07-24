#!/usr/bin/env python3
"""Atomically archive the exact passed system, code, weights, and submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model  # noqa: E402


SCHEMA = "poke_bot.passed_system_elmo_archive/v1"
EXCLUDED_PREFIXES = (
    ".git/",
    "outputs/",
    "data/episodes/",
    "data/bootstrap/",
    "node_modules/",
    ".venv/",
    "venv/",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(argv: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {shlex.join(argv)}\n"
            + (completed.stdout or "")
        )
    return completed.stdout or ""


def _safe_source_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    selected: list[Path] = []
    for value in raw.split(b"\0"):
        if not value:
            continue
        relative = Path(os.fsdecode(value))
        normalized = relative.as_posix()
        if normalized.startswith(EXCLUDED_PREFIXES):
            continue
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            continue
        if source.is_file() or source.is_symlink():
            selected.append(relative)
    return sorted(set(selected), key=lambda value: value.as_posix())


def _tar_relative(output: Path, root: Path, paths: Iterable[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", dereference=False) as archive:
        for relative in paths:
            source = root / relative
            if source.exists() or source.is_symlink():
                archive.add(source, arcname=relative.as_posix(), recursive=True)


def _write_runtime_contracts(stage: Path, source_root: Path, services: list[str]) -> None:
    contracts = stage / "runtime-contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    for service in services:
        for suffix, argv in (
            ("cat", ["systemctl", "--user", "cat", service]),
            (
                "show",
                [
                    "systemctl",
                    "--user",
                    "show",
                    service,
                    "-p",
                    "FragmentPath",
                    "-p",
                    "DropInPaths",
                    "-p",
                    "Environment",
                    "-p",
                    "ExecStart",
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "NRestarts",
                ],
            ),
        ):
            completed = subprocess.run(
                argv,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            # Some components (notably the dashboard) intentionally live on a
            # different host. Preserve the absence as evidence instead of
            # aborting an otherwise complete model archive.
            (contracts / f"{service}.{suffix}.txt").write_text(
                f"returncode={completed.returncode}\n{completed.stdout or ''}",
                encoding="utf-8",
            )
    for name, argv in {
        "git-head.txt": ["git", "rev-parse", "HEAD"],
        "git-status.txt": ["git", "status", "--porcelain=v2", "--branch"],
        "git-diff.patch": ["git", "diff", "--binary", "HEAD"],
        "git-diff-cached.patch": ["git", "diff", "--binary", "--cached"],
        "python-environment.txt": [
            "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python",
            "-m",
            "pip",
            "freeze",
        ],
    }.items():
        (contracts / name).write_text(
            command(argv, cwd=source_root, capture=True), encoding="utf-8"
        )


def _validate_handler(state: dict[str, Any], frozen: dict[str, Any]) -> None:
    attempts = [row for row in state.get("submission_attempts") or [] if isinstance(row, dict)]
    slots = {int(row.get("slot", -1)) for row in attempts if row.get("attempted") is True}
    gate = dict(state.get("gate") or {})
    bundle = dict(state.get("submission_bundle") or {})
    if (
        state.get("phase")
        not in {
            "two_submissions_attempted",
            "waiting_for_terminal_trainer_before_handoff",
            "complete_handoff_started",
        }
        or state.get("approved_submission_count") != 2
        or slots != {1, 2}
        or state.get("automatic_retries") is not False
        or gate.get("checkpoint_digest") != frozen.get("checkpoint_digest")
        or bundle.get("sha256") is None
    ):
        raise RuntimeError("handler state is not a verified exact-pass/two-attempt handoff")


def _hash_rows(stage: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"FILES.sha256", "ARCHIVE_MANIFEST.json"}:
            rows.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handler-state", type=Path, required=True)
    parser.add_argument("--frozen-family", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--destination-host", default="elmo")
    parser.add_argument(
        "--destination-root", default="/mnt/Main/main/poke-model-archive"
    )
    parser.add_argument(
        "--elmo-worker-deployment",
        default="/mnt/Main/main/poke-bot-agent/deployments/persistent-workers-20260720-v1",
    )
    parser.add_argument(
        "--elmo-image-archive-source",
        default=(
            "/mnt/Main/main/poke-model-archive/alakazam-baseline-gate-d02df/elmo/"
            "poke-bot-truenas-worker_safety-20260720.8-persistent-workers.docker.tar.zst"
        ),
    )
    args = parser.parse_args()

    frozen = verify_frozen_model(args.frozen_family.expanduser().resolve())
    handler = read_json(args.handler_state.expanduser().resolve())
    _validate_handler(handler, frozen)
    digest_short = str(frozen["checkpoint_digest"]).removeprefix("sha256:")[:16]
    archive_id = f"alakazam-strong-public-gate-{digest_short}"
    final_remote = f"{args.destination_root.rstrip('/')}/{archive_id}"
    if args.receipt.is_file():
        receipt = read_json(args.receipt)
        if (
            receipt.get("schema") == SCHEMA
            and receipt.get("archive_id") == archive_id
            and receipt.get("destination") == final_remote
            and receipt.get("checkpoint_digest") == frozen["checkpoint_digest"]
        ):
            command(
                [
                    "ssh",
                    args.destination_host,
                    "test",
                    "-s",
                    f"{final_remote}/ARCHIVE_MANIFEST.json",
                ]
            )
            print(json.dumps(receipt, indent=2), flush=True)
            return 0
        raise RuntimeError("existing archive receipt identity differs")

    stage = args.staging_root.expanduser().resolve() / f".{archive_id}.partial"
    if stage.exists():
        raise FileExistsError(f"stale archive staging directory: {stage}")
    stage.mkdir(parents=True)
    source_root = args.source_root.expanduser().resolve()
    deployment_root = args.deployment_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    submission_root = args.submission_root.expanduser().resolve()

    _write_runtime_contracts(
        stage,
        source_root,
        [
            "pokebot-pure-rl-alakazam.service",
            "pokebot-passed-gate-handler.service",
            "pokebot-gate-aligned-learner-v18-boundary.service",
            "pokebot-dashboard.service",
        ],
    )
    _tar_relative(stage / "source-tree.tar.gz", source_root, _safe_source_files(source_root))
    _tar_relative(
        stage / "active-deployment.tar.gz",
        deployment_root.parent,
        [Path(deployment_root.name)],
    )
    run_paths = [
        Path(name)
        for name in (
            "manifest.json",
            "loop_state.json",
            "SPECIALIST_GATE_PASSED",
            "commits",
            "eval",
            "metrics",
            "checkpoints",
            "collection_receipts",
            "rehearsals",
        )
    ]
    _tar_relative(stage / "gate-run-evidence-and-weights.tar.gz", run_dir, run_paths)
    _tar_relative(
        stage / "protected-frozen-model.tar.gz",
        args.frozen_family.expanduser().resolve().parent,
        [Path(args.frozen_family.expanduser().resolve().name)],
    )
    _tar_relative(
        stage / "submission-bundles-and-receipts.tar.gz",
        submission_root.parent,
        [Path(submission_root.name)],
    )
    engine = args.engine.expanduser().resolve()
    if not engine.is_file():
        raise FileNotFoundError(engine)
    _tar_relative(stage / "engine.tar.gz", engine.parent, [Path(engine.name)])
    atomic_json(stage / "handler-state.json", handler)

    files = _hash_rows(stage)
    manifest = {
        "schema": SCHEMA,
        "archive_id": archive_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_digest": frozen["checkpoint_digest"],
        "frozen_model": frozen,
        "gate": handler.get("gate"),
        "submission_attempts": handler.get("submission_attempts"),
        "submission_bundle": handler.get("submission_bundle"),
        "source_snapshot_includes_dirty_and_untracked": True,
        "large_training_data_and_replay_shards_excluded": True,
        "files": files,
    }
    atomic_json(stage / "ARCHIVE_MANIFEST.json", manifest)
    final_rows = _hash_rows(stage)
    (stage / "FILES.sha256").write_text(
        "".join(
            f"{row['sha256'].removeprefix('sha256:')}  {row['path']}\n"
            for row in final_rows
        ),
        encoding="utf-8",
    )

    remote_partial = f"{args.destination_root.rstrip('/')}/.{archive_id}.partial"
    remote_init = (
        f"set -eu; test ! -e {shlex.quote(final_remote)}; "
        f"rm -rf {shlex.quote(remote_partial)}; mkdir -p {shlex.quote(remote_partial)}"
    )
    command(["ssh", args.destination_host, remote_init])
    command(
        [
            "rsync",
            "-a",
            "--partial",
            str(stage) + "/",
            f"{args.destination_host}:{remote_partial}/",
        ]
    )
    remote_worker = shlex.quote(str(args.elmo_worker_deployment))
    remote_image = shlex.quote(str(args.elmo_image_archive_source))
    remote_finalize = f"""
set -eu
P={shlex.quote(remote_partial)}
F={shlex.quote(final_remote)}
cd "$P"
sha256sum -c FILES.sha256
mkdir -p elmo
sudo -n docker inspect poke-bot-truenas-worker > elmo/worker-container-inspect.json
sudo -n docker image inspect poke-bot-truenas-worker:safety-20260720.8-persistent-workers > elmo/worker-image-inspect.json
tar -C $(dirname {remote_worker}) -czf elmo/worker-deployment.tar.gz $(basename {remote_worker})
if test -s {remote_image}; then
  ln {remote_image} elmo/poke-bot-truenas-worker_safety-20260720.8-persistent-workers.docker.tar.zst 2>/dev/null || cp --reflink=auto {remote_image} elmo/poke-bot-truenas-worker_safety-20260720.8-persistent-workers.docker.tar.zst
fi
find . -type f ! -name FINAL_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > FINAL_SHA256SUMS
sha256sum -c FINAL_SHA256SUMS
printf '%s\n' {shlex.quote(frozen['checkpoint_digest'])} > VERIFIED_CHECKPOINT_DIGEST
cd ..
mv "$P" "$F"
"""
    command(["ssh", args.destination_host, remote_finalize])

    receipt = {
        "schema": SCHEMA,
        "status": "verified",
        "archive_id": archive_id,
        "destination_host": args.destination_host,
        "destination": final_remote,
        "checkpoint_digest": frozen["checkpoint_digest"],
        "local_manifest_sha256": sha256(stage / "ARCHIVE_MANIFEST.json"),
        "submitted_slots": [1, 2],
        "automatic_retries": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
