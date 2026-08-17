"""Focused immutable-source checks for the r219 local mirror snapshotter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT
    / "scripts/stage_alakazam_local_multi_search_turn_belief_mcts_r219_source_snapshot.py"
)


@pytest.fixture()
def snapshot_module():
    spec = importlib.util.spec_from_file_location("r219_source_snapshot_test", SNAPSHOT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_r218_input(
    snapshot_module, root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, bytes]:
    """Make a small physical r218-shaped input and patch only test identities."""

    payloads = {
        "model": b"frozen r195 model for snapshot test\n",
        "tree": b'{"runtime_enabled":true,"tree":"r195"}\n',
        "engine": b"B77 engine test bytes\n",
        "deck": b"1,a\n" * 60,
        "agent": b"# archived r195 agent\n",
        "mcts": b"# archived r195 mcts\n",
        "init": b"# archived r195 init\n",
    }
    model_sha = "sha256:" + hashlib.sha256(payloads["model"]).hexdigest()
    tree_sha = "sha256:" + hashlib.sha256(payloads["tree"]).hexdigest()
    engine_sha = "sha256:" + hashlib.sha256(payloads["engine"]).hexdigest()
    monkeypatch.setattr(snapshot_module, "R195_CHECKPOINT_SHA256", model_sha)
    monkeypatch.setattr(snapshot_module, "R195_MATCHUP_TREE_SHA256", tree_sha)
    monkeypatch.setattr(snapshot_module, "B77_ENGINE_SHA256", engine_sha)
    monkeypatch.setattr(
        snapshot_module,
        "R195_ARCHIVED_AGENT_SHA256",
        "sha256:" + hashlib.sha256(payloads["agent"]).hexdigest(),
    )
    monkeypatch.setattr(
        snapshot_module,
        "R195_ARCHIVED_MCTS_SHA256",
        "sha256:" + hashlib.sha256(payloads["mcts"]).hexdigest(),
    )
    monkeypatch.setattr(
        snapshot_module,
        "R195_ARCHIVED_PACKAGE_INIT_SHA256",
        "sha256:" + hashlib.sha256(payloads["init"]).hexdigest(),
    )
    monkeypatch.setattr(
        snapshot_module, "R218_CONTRACT_SHA256", "sha256:test-r218-contract"
    )

    for package in snapshot_module.FROZEN_PACKAGE_NAMES:
        package_root = root / package
        (package_root / "cg").mkdir(parents=True)
        (package_root / "poke_bot").mkdir()
        (package_root / "poke_bot/__init__.py").write_bytes(payloads["init"])
        (package_root / "poke_bot/agent.py").write_bytes(payloads["agent"])
        (package_root / "poke_bot/mcts.py").write_bytes(payloads["mcts"])
        (package_root / "main.py").write_text("# archived main\n", encoding="utf-8")
        (package_root / "model.pt").write_bytes(payloads["model"])
        (package_root / "deck.csv").write_bytes(payloads["deck"])
        (package_root / "matchup_tree.json").write_bytes(payloads["tree"])
        (package_root / "runtime_profile.json").write_text("{}\n", encoding="utf-8")
        (package_root / "turn_order_profile.json").write_text("{}\n", encoding="utf-8")
        (package_root / "cg/__init__.py").write_text("# cg\n", encoding="utf-8")
        (package_root / "cg/api.py").write_text("# api\n", encoding="utf-8")
        (package_root / "cg/game.py").write_text("# game\n", encoding="utf-8")
        (package_root / "cg/sim.py").write_text("# sim\n", encoding="utf-8")
        (package_root / "cg/libcg.so").write_bytes(payloads["engine"])

    manifest = {
        "schema": "poke_bot.r218_local_first_decision_bo1000_source_manifest/v1",
        "owner_decision_revision": 218,
        "contract_sha256": snapshot_module.R218_CONTRACT_SHA256,
        "r195_checkpoint_sha256": model_sha,
        "r195_matchup_tree_sha256": tree_sha,
        "seeded_engine_sha256": engine_sha,
    }
    manifest_path = root / snapshot_module.R218_INPUT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        snapshot_module, "R218_INPUT_MANIFEST_SHA256", _sha256(manifest_path)
    )
    return payloads


def _write_runtime_source(
    snapshot_module, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = root / "poke_bot"
    package.mkdir(parents=True)
    for relative in snapshot_module.OVERLAY_POKE_BOT_RELATIVES:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# selected r219 overlay {relative.as_posix()}\n",
            encoding="utf-8",
        )
    # The complete snapshot also retains an arbitrary non-overlay module.
    (package / "ordinary_runtime_dependency.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ordinary_runtime_dependency.cpython-311.pyc").write_bytes(b"stale")

    runner = root / snapshot_module.RUNNER_RELATIVE
    runner.parent.mkdir(parents=True)
    runner.write_text("# r219 runner\n", encoding="utf-8")
    contract = root / snapshot_module.R219_CONTRACT_RELATIVE
    contract.parent.mkdir(parents=True)
    contract.write_bytes((ROOT / snapshot_module.R219_CONTRACT_RELATIVE).read_bytes())

    selected = {
        key: _sha256(root / key) for key in snapshot_module.SELECTED_RUNTIME_CODE_SHA256
    }
    monkeypatch.setattr(snapshot_module, "SELECTED_RUNTIME_CODE_SHA256", selected)


def _preflight_receipt(*_args, **_kwargs) -> dict[str, object]:
    return {
        "schema": "poke_bot.r219_source_snapshot_env_i_preflight/v1",
        "python": "3.11.15",
        "runtime_modules": {},
        "runner": "scripts/run_alakazam_local_multi_search_turn_belief_mcts_bo1000_r219.py",
        "frozen_package_engines": {},
    }


def _stage(snapshot_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    r218_input = tmp_path / "r218-input"
    r218_input.mkdir()
    payloads = _write_r218_input(snapshot_module, r218_input, monkeypatch)
    runtime = tmp_path / "runtime-source"
    runtime.mkdir()
    _write_runtime_source(snapshot_module, runtime, monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(snapshot_module, "_run_sanitized_preflight", _preflight_receipt)
    result = snapshot_module.stage_snapshot(
        r218_input_root=r218_input,
        runtime_source_root=runtime,
        staging_parent=staging,
        python=Path(sys.executable),
    )
    return result, payloads, r218_input, runtime


def test_stage_creates_a_fresh_sealed_physical_closure_with_both_package_overlays(
    snapshot_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, payloads, _input, runtime = _stage(snapshot_module, tmp_path, monkeypatch)
    source_root = Path(result["source_root"])
    assert result["status"] == "sealed"
    assert source_root.name.startswith(snapshot_module.DEPLOYMENT_PREFIX)
    assert stat.S_IMODE(source_root.lstat().st_mode) == 0o555

    manifest = json.loads((source_root / snapshot_module.MANIFEST_NAME).read_text())
    assert manifest["source_tree_sha256"] == result["source_tree_sha256"]
    assert manifest["r219_contract"]["sha256"] == snapshot_module.R219_CONTRACT_SHA256
    assert (
        manifest["r218_input"]["r218_manifest"]["sha256"]
        == snapshot_module.R218_INPUT_MANIFEST_SHA256
    )
    assert manifest["frozen_r195_inputs"]["direct_and_mcts_copied_physically"]
    assert (
        "poke_bot/r219_seeded_mirror_runtime.py"
        in manifest["package_namespace_overlays"]["overlaid_poke_bot_relatives"]
    )

    for package in snapshot_module.FROZEN_PACKAGE_NAMES:
        assert (source_root / package / "model.pt").read_bytes() == payloads["model"]
        assert (source_root / package / "matchup_tree.json").read_bytes() == payloads[
            "tree"
        ]
        assert (source_root / package / "cg/libcg.so").read_bytes() == payloads[
            "engine"
        ]
        assert (source_root / package / "poke_bot/agent.py").read_bytes() == payloads[
            "agent"
        ]
        assert (source_root / package / "poke_bot/mcts.py").read_bytes() == payloads[
            "mcts"
        ]
        assert (
            source_root / package / "poke_bot/__init__.py"
        ).read_bytes() == payloads["init"]
        for relative in snapshot_module.OVERLAY_POKE_BOT_RELATIVES:
            staged = source_root / package / "poke_bot" / relative
            selected = runtime / "poke_bot" / relative
            assert staged.read_bytes() == selected.read_bytes()
            assert stat.S_IMODE(staged.lstat().st_mode) == 0o444

    assert (source_root / "poke_bot/ordinary_runtime_dependency.py").is_file()
    assert not (source_root / "poke_bot/__pycache__").exists()
    assert snapshot_module.verify_snapshot(source_root)["status"] == "passed"


def test_stage_rejects_a_symlinked_frozen_package_input(
    snapshot_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r218_input = tmp_path / "r218-input"
    r218_input.mkdir()
    _write_r218_input(snapshot_module, r218_input, monkeypatch)
    model = r218_input / "direct/model.pt"
    model.unlink()
    model.symlink_to(r218_input / "mcts/model.pt")
    runtime = tmp_path / "runtime-source"
    runtime.mkdir()
    _write_runtime_source(snapshot_module, runtime, monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(snapshot_module.SnapshotError, match="physical regular file"):
        snapshot_module.stage_snapshot(
            r218_input_root=r218_input,
            runtime_source_root=runtime,
            staging_parent=staging,
            python=Path(sys.executable),
        )


def test_verify_rejects_critical_runtime_drift_after_seal(
    snapshot_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _payloads, _input, _runtime = _stage(snapshot_module, tmp_path, monkeypatch)
    source_root = Path(result["source_root"])
    changed = source_root / "direct/poke_bot/r219_multi_search_turn_belief_mcts.py"
    changed.chmod(0o644)
    changed.write_text("# drift\n", encoding="utf-8")

    with pytest.raises(snapshot_module.SnapshotError, match="package overlay drifted"):
        snapshot_module.verify_snapshot(source_root)


def test_runtime_preflight_uses_a_sanitized_env_i_style_environment(
    snapshot_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "sealed"
    source_root.mkdir()
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "schema": "poke_bot.r219_source_snapshot_env_i_preflight/v1",
                    "python": "3.11.9",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(snapshot_module.subprocess, "run", fake_run)
    receipt = snapshot_module._run_sanitized_preflight(
        source_root, Path(sys.executable), timeout_seconds=12.0
    )
    assert receipt["python"] == "3.11.9"
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert environment == {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "PATH": snapshot_module.os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(source_root),
        "R219_SOURCE_ROOT": str(source_root),
    }
    assert kwargs["cwd"] == source_root
    assert kwargs["timeout"] == 12.0


def test_stage_refuses_a_parent_inside_an_input_tree(
    snapshot_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r218_input = tmp_path / "r218-input"
    r218_input.mkdir()
    _write_r218_input(snapshot_module, r218_input, monkeypatch)
    runtime = tmp_path / "runtime-source"
    runtime.mkdir()
    _write_runtime_source(snapshot_module, runtime, monkeypatch)
    nested = r218_input / "forbidden-stage-parent"
    nested.mkdir()

    with pytest.raises(
        snapshot_module.SnapshotError, match="may not be inside either input tree"
    ):
        snapshot_module.stage_snapshot(
            r218_input_root=r218_input,
            runtime_source_root=runtime,
            staging_parent=nested,
            python=Path(sys.executable),
        )
