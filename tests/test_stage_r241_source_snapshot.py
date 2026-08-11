from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage_r241_source_snapshot.py"


def _module():
    spec = importlib.util.spec_from_file_location("r241_snapshot_stager_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r241_source_snapshot_stager_publishes_a_readonly_full_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    stager = _module()
    # A content-addressed r241 root is published by exclusive mkdir, never a
    # rename that could replace a concurrent empty directory on macOS.
    monkeypatch.setattr(
        stager.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rename used")),
    )
    output_base = tmp_path / "deployments"
    outputs_root = tmp_path / "outputs"
    receipt = outputs_root / "state/source-snapshot-staging.json"
    output_base.mkdir()
    receipt.parent.mkdir(parents=True)

    first = stager._stage(
        launcher=stager._load_launcher(),
        output_base=output_base,
        outputs_root=outputs_root,
        receipt_output=receipt,
        host="inzi",
    )
    second = stager._stage(
        launcher=stager._load_launcher(),
        output_base=output_base,
        outputs_root=outputs_root,
        receipt_output=receipt,
        host="inzi",
    )

    assert second == first
    snapshot = dict(first["source_snapshot"])
    snapshot_root = Path(str(snapshot["root"]))
    manifest = Path(str(snapshot["manifest"]))
    assert snapshot_root.parent == output_base.resolve()
    assert snapshot_root.name == (
        "alakazam-new-list-direct-r241-src-"
        + str(snapshot["manifest_sha256"]).removeprefix("sha256:")[:16]
    )
    assert snapshot["outputs_root"] == str(outputs_root.resolve())
    assert snapshot_root.stat().st_mode & stat.S_IWUSR == 0
    assert manifest.stat().st_mode & stat.S_IWUSR == 0
    assert not (snapshot_root / "outputs").exists()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {row["path"] for row in payload["files"]}
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
    assert {
        "scripts/stage_r241_source_snapshot.py",
        "scripts/canary_game_accuracy.py",
        "scripts/resource_watcher.py",
        "scripts/unattended_monitor.py",
        "poke_bot/own_deck_successor.py",
        "state/alakazam-own-deck-ledger-successor-r258.json",
        "deploy/elmo/docker-compose.r241-elmo-official-r236-remote-worker.yml.template",
        "deploy/elmo/r241-elmo-official-r236-remote-worker.env.template",
        "deploy/systemd/pokebot-r241-elmo-official-r236-remote-worker.service.template",
    }.issubset(paths)
    sealed_r258 = json.loads(
        (
            snapshot_root
            / "state/alakazam-own-deck-ledger-successor-r258.json"
        ).read_text(encoding="utf-8")
    )
    assert sealed_r258["schema"] == "poke_bot.alakazam_own_deck_ledger_successor_r258/v1"
    assert sealed_r258["canonical"] is True
    assert set(stager._load_launcher()._REQUIRED_SOURCE_SNAPSHOT_FILES).issubset(paths)
    # Every descendant directory is sealed too: a read-only root alone would
    # still permit an interpreter to create an unbound ``__pycache__`` under a
    # writable package directory after publication.
    assert all(path.stat().st_mode & 0o222 == 0 for path in snapshot_root.rglob("*"))
    assert not any(
        path.is_dir() and path.name == "__pycache__" for path in snapshot_root.rglob("*")
    )

    staged_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert staged_receipt["status"] == "passed"
    assert staged_receipt["source_snapshot"] == snapshot


def test_r241_source_snapshot_stager_refuses_a_receipt_outside_external_outputs(
    tmp_path: Path,
) -> None:
    stager = _module()
    output_base = tmp_path / "deployments"
    outputs_root = tmp_path / "outputs"
    output_base.mkdir()
    outputs_root.mkdir()

    with pytest.raises(stager.R241SourceSnapshotError, match="under external outputs"):
        stager._stage(
            launcher=stager._load_launcher(),
            output_base=output_base,
            outputs_root=outputs_root,
            receipt_output=tmp_path / "outside.json",
            host="inzi",
        )


def test_r241_source_snapshot_verifier_emits_a_host_receipt_from_the_sealed_root(
    tmp_path: Path,
) -> None:
    stager = _module()
    output_base = tmp_path / "deployments"
    outputs_root = tmp_path / "outputs"
    output_base.mkdir()
    (outputs_root / "state").mkdir(parents=True)
    staged = stager._stage(
        launcher=stager._load_launcher(),
        output_base=output_base,
        outputs_root=outputs_root,
        receipt_output=outputs_root / "state/initial-stage.json",
        host="inzi",
    )
    snapshot = dict(staged["source_snapshot"])
    snapshot_root = Path(str(snapshot["root"]))
    receipt = outputs_root / "state/published-elmo-stage.json"
    command = [
        sys.executable,
        "-B",
        str(snapshot_root / "scripts/stage_r241_source_snapshot.py"),
        "--published-root",
        str(snapshot_root),
        "--outputs-root",
        str(outputs_root),
        "--receipt-output",
        str(receipt),
        "--host",
        "elmo",
    ]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    first = subprocess.run(
        command,
        cwd=snapshot_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    second = subprocess.run(
        command,
        cwd=snapshot_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    result = json.loads(first.stdout)
    assert result == json.loads(second.stdout)

    # The deployment command supplies both `-B` and
    # `PYTHONDONTWRITEBYTECODE=1`, but the copied verifier independently
    # hardens itself before loading its launcher.  A NAS ACL must not turn an
    # omitted caller flag into a newly created, unbound cache directory.
    unguarded_command = [sys.executable, *command[2:]]
    unguarded_environment = dict(os.environ)
    unguarded_environment.pop("PYTHONDONTWRITEBYTECODE", None)
    third = subprocess.run(
        unguarded_command,
        cwd=snapshot_root,
        env=unguarded_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert third.returncode == 0, third.stderr
    assert result == json.loads(third.stdout)

    launcher_check = subprocess.run(
        [
            sys.executable,
            str(snapshot_root / "scripts/launch_alakazam_new_list_direct_r241.py"),
            "--static-check",
        ],
        cwd=snapshot_root,
        env=unguarded_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert launcher_check.returncode == 0, launcher_check.stderr
    assert result["operation"] == "verify_published_immutable_source_snapshot"
    assert result["source_snapshot"]["host"] == "elmo"
    assert result["source_snapshot"]["root"] == str(snapshot_root)
    assert result["source_snapshot"]["manifest_sha256"] == snapshot["manifest_sha256"]
    assert result["source_snapshot"]["source_tree_sha256"] == snapshot["source_tree_sha256"]
    assert receipt.stat().st_mode & stat.S_IWUSR == 0
    assert snapshot_root.stat().st_mode & stat.S_IWUSR == 0
    assert not any(
        path.is_dir() and path.name == "__pycache__" for path in snapshot_root.rglob("*")
    )

    # A mutable checkout cannot issue an attestation for some other source
    # root; verification has to run from the sealed bytes it certifies.
    with pytest.raises(
        stager.R241SourceSnapshotError,
        match="must execute from that immutable source root",
    ):
        stager._verify_published(
            launcher=stager._load_launcher(),
            snapshot_root=snapshot_root,
            outputs_root=outputs_root,
            receipt_output=outputs_root / "state/rejected-mutable-verifier.json",
            host="inzi",
        )
