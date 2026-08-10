from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/stage_r228_async_eight_worker_kaggle_viability.py"


def _load_stage_module():
    spec = importlib.util.spec_from_file_location("r228_stage_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in sorted(files.items()):
            source = path.parent / ("source-" + name.replace("/", "-"))
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(body)
            archive.add(source, arcname="./" + name)


def _copy_wrapper_sources(destination: Path) -> None:
    for relative in (
        "submission/r228_async_eight_worker_main.py",
        "poke_bot/r228_kaggle_async_runtime.py",
        "poke_bot/r228_async_shared_tree_queue.py",
        "poke_bot/r225_stock_native_lane.py",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_r228_stager_is_deterministic_and_preserves_frozen_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)

    files = {
        "main.py": b"def agent(obs):\n    return [0]\n",
        "model.pt": b"r195-model",
        "matchup_tree.json": b"r195-matchup-tree",
        "search_config.json": b'{"frozen":true}\n',
        "cg/libcg.so": b"stock-libcg",
    }
    archive = tmp_path / "r195.tar.gz"
    _write_archive(archive, files)
    monkeypatch.setattr(stage_mod, "R195_BUNDLE_SHA256", _sha_bytes(archive.read_bytes()))
    monkeypatch.setattr(stage_mod, "R195_MODEL_SHA256", _sha_bytes(files["model.pt"]))
    monkeypatch.setattr(
        stage_mod, "R195_MATCHUP_TREE_SHA256", _sha_bytes(files["matchup_tree.json"])
    )
    monkeypatch.setattr(
        stage_mod, "R195_SEARCH_CONFIG_SHA256", _sha_bytes(files["search_config.json"])
    )
    monkeypatch.setattr(stage_mod, "STOCK_LIBCG_SHA256", _sha_bytes(files["cg/libcg.so"]))
    monkeypatch.setattr(stage_mod, "STOCK_LIBCG_BYTES", len(files["cg/libcg.so"]))

    first = stage_mod.stage_bundle(
        r195_bundle=archive, output_dir=tmp_path / "one", source_root=source_root
    )
    second = stage_mod.stage_bundle(
        r195_bundle=archive, output_dir=tmp_path / "two", source_root=source_root
    )
    assert first == second
    assert first["kaggle_submission_created"] is False
    assert first["async_selected_action_authority"] == "receipt.selected_action"

    staged = tmp_path / "one" / stage_mod.ARCHIVE_FILENAME
    with tarfile.open(staged, "r:gz") as archive_file:
        names = {name.removeprefix("./") for name in archive_file.getnames()}
        assert {
            "main.py",
            "r195_direct_main.py",
            "model.pt",
            "matchup_tree.json",
            "search_config.json",
            "cg/libcg.so",
            "poke_bot/r228_kaggle_async_runtime.py",
            "poke_bot/r228_async_shared_tree_queue.py",
            "poke_bot/r225_stock_native_lane.py",
            stage_mod.MANIFEST_FILENAME,
        } <= names
        search_config = archive_file.extractfile("./search_config.json")
        assert search_config is not None
        assert search_config.read() == files["search_config.json"]
        manifest_file = archive_file.extractfile("./" + stage_mod.MANIFEST_FILENAME)
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read())
        assert manifest["entrypoint_sha256"] == first["entrypoint_sha256"]
        assert manifest["required_label"] == stage_mod.REQUIRED_LABEL


def test_r228_stage_rejects_a_direct_policy_side_probe(tmp_path: Path) -> None:
    stage_mod = _load_stage_module()
    wrapper = tmp_path / "main.py"
    runtime = tmp_path / "runtime.py"
    wrapper.write_text(
        "R228_ASYNC_SELECTED_ACTION_AUTHORITY = 'receipt.selected_action'\n"
        "from poke_bot import r228_kaggle_async_runtime\n"
        "def agent(obs):\n    return [0]\n",
        encoding="utf-8",
    )
    runtime.write_text(
        "DECISION_PREFIX = 'R228_ASYNC_EIGHT_WORKER_DECISION'\n"
        "def select():\n"
        "    receipt = search.run_decision()\n"
        "    return [0]\n",
        encoding="utf-8",
    )
    with pytest.raises(stage_mod.R228StageError, match="selected action"):
        stage_mod.validate_async_action_authority(wrapper, runtime)


def test_r228_stage_rejects_main_that_uses_search_as_a_side_probe(tmp_path: Path) -> None:
    stage_mod = _load_stage_module()
    wrapper = tmp_path / "main.py"
    runtime = tmp_path / "runtime.py"
    wrapper.write_text(
        "R228_ASYNC_SELECTED_ACTION_AUTHORITY = 'receipt.selected_action'\n"
        "from poke_bot import r228_kaggle_async_runtime\n"
        "def agent(obs):\n    return [0]\n",
        encoding="utf-8",
    )
    runtime.write_text(
        "DECISION_PREFIX = 'R228_ASYNC_EIGHT_WORKER_DECISION'\n"
        "def select():\n"
        "    receipt = search.run_decision()\n"
        "    selected = receipt.selected_action\n"
        "    return list(selected)\n",
        encoding="utf-8",
    )
    with pytest.raises(stage_mod.R228StageError, match="does not return its r228 runtime"):
        stage_mod.validate_async_action_authority(wrapper, runtime)
