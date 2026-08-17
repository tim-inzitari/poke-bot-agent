from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from poke_bot.r225_eight_lane_diagnostic import _RootEdge, _SharedDiagnosticTree


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/stage_alakazam_r225_eight_lane_shared_tree_diagnostic.py"


def _load_stage_module():
    spec = importlib.util.spec_from_file_location("r225_stage_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in sorted(files.items()):
            source = path.parent / ("source-" + name.replace("/", "-"))
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(body)
            archive.add(source, arcname="./" + name)


def _copy_bundle_sources(destination: Path) -> None:
    for relative in (
        "submission/r225_eight_lane_diagnostic_main.py",
        "poke_bot/r225_eight_lane_diagnostic.py",
        "poke_bot/r225_stock_native_lane.py",
        "poke_bot/r222_stock_shared_tree_batch.py",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_shared_tree_reservations_are_transactional_and_never_merge_lookalikes() -> None:
    tree = _SharedDiagnosticTree(
        [
            _RootEdge((0,), 0.6),
            _RootEdge((1,), 0.3),
            _RootEdge((2,), 0.1),
        ]
    )
    tokens = [tree.reserve() for _ in range(8)]
    assert tree.outstanding_reservations == 8
    assert tree.unsafe_public_lookalike_merges == 0
    assert tree.cache_hits == 0
    undos = [tree.backup(token, 0.25) for token in tokens]
    assert tree.outstanding_reservations == 0
    assert tree.total_backups == 8
    # A coordinator failure after complete backups can reverse the exact tree
    # changes, then abort without retaining virtual loss.
    for undo in reversed(undos):
        undo()
    tree.abort(RuntimeError("test rollback"))
    assert tree.outstanding_reservations == 0
    assert tree.total_backups == 0
    assert all(edge.visits == 0 and edge.reservations == 0 for edge in tree.edges)


def test_r225_stage_keeps_r195_search_config_byte_identical_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_bundle_sources(source_root)
    stage_mod.ROOT = source_root

    main = b"def agent(obs):\n    return [0]\n"
    model = b"r195-model"
    tree = b"r195-matchup-tree"
    search_config = b'{"enabled":false,"frozen":true}\n'
    libcg = b"stock-libcg"
    archive = tmp_path / "r195.tar.gz"
    _write_archive(
        archive,
        {
            "main.py": main,
            "model.pt": model,
            "matchup_tree.json": tree,
            "search_config.json": search_config,
            "turn_order_profile.json": json.dumps(
                {
                    "schema": "poke_bot.submission_turn_order_profile/v1",
                    "turn_order_preference": "first_if_allowed",
                }
            ).encode(),
            "cg/libcg.so": libcg,
            "runtime_profile.json": json.dumps(
                {
                    "schema": "poke_bot.submission_runtime_profile/v1",
                    "recursive_turn_planner": "disabled",
                    "display": "NO RTP",
                    "rtp_sidecar_packaged": False,
                }
            ).encode(),
        },
    )
    r222_contract = source_root / "state/r222.json"
    r222_contract.parent.mkdir(parents=True)
    r222_contract.write_text("{}\n", encoding="utf-8")
    fake_r222_sha = _sha(r222_contract)
    contract = source_root / "state/r225.json"
    contract_payload = {
        "schema": stage_mod.SCHEMA,
        "owner_decision_revision": 225,
        "exact_frozen_base": {
            "r195_bundle_sha256": _sha(archive),
            "r195_checkpoint_sha256": "sha256:" + hashlib.sha256(model).hexdigest(),
            "r195_matchup_tree_sha256": "sha256:" + hashlib.sha256(tree).hexdigest(),
            "stock_libcg_sha256": "sha256:" + hashlib.sha256(libcg).hexdigest(),
            "stock_libcg_size_bytes": len(libcg),
        },
        "relationship_to_existing_work": {
            "r222_contract_path": "state/r222.json",
            "r222_contract_sha256": fake_r222_sha,
        },
        "authority": {
            "kaggle_api_call_permitted_now_before_preconditions": False,
            "kaggle_upload_permitted_now_before_preconditions": False,
            "kaggle_queue_submission_permitted": False,
            "automatic_kaggle_submission_allowed": False,
        },
    }
    contract.write_text(json.dumps(contract_payload), encoding="utf-8")
    fake_contract_sha = _sha(contract)
    monkeypatch.setattr(stage_mod, "R195_BUNDLE_SHA256", _sha(archive))
    monkeypatch.setattr(stage_mod, "R195_MODEL_SHA256", contract_payload["exact_frozen_base"]["r195_checkpoint_sha256"])
    monkeypatch.setattr(stage_mod, "R195_MATCHUP_TREE_SHA256", contract_payload["exact_frozen_base"]["r195_matchup_tree_sha256"])
    monkeypatch.setattr(stage_mod, "STOCK_LIBCG_SHA256", contract_payload["exact_frozen_base"]["stock_libcg_sha256"])
    monkeypatch.setattr(stage_mod, "STOCK_LIBCG_BYTES", len(libcg))
    monkeypatch.setattr(stage_mod, "SEARCH_CONFIG_SHA256", "sha256:" + hashlib.sha256(search_config).hexdigest())
    monkeypatch.setattr(stage_mod, "R222_CONTRACT_SHA256", fake_r222_sha)
    monkeypatch.setattr(stage_mod, "R225_CONTRACT_SHA256", fake_contract_sha)

    first = stage_mod.stage_bundle(
        r195_bundle=archive,
        contract_path=contract,
        output_dir=tmp_path / "one",
        source_root=source_root,
        budget_s=4.0,
        batches=2,
    )
    second = stage_mod.stage_bundle(
        r195_bundle=archive,
        contract_path=contract,
        output_dir=tmp_path / "two",
        source_root=source_root,
        budget_s=4.0,
        batches=2,
    )
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["kaggle_api_called"] is False
    with tarfile.open(Path(first["bundle"]), "r:gz") as staged:
        names = {_name.removeprefix("./") for _name in staged.getnames()}
        assert {
            "main.py",
            "r195_direct_main.py",
            "r225_eight_lane_diagnostic_config.json",
            "R225_EIGHT_LANE_DIAGNOSTIC_README.md",
            "r225_eight_lane_diagnostic_manifest.json",
            "contracts/r225-typed-contract.json",
            "contracts/r222-typed-contract.json",
        } <= names
        config_member = staged.extractfile("./search_config.json")
        assert config_member is not None
        assert config_member.read() == search_config
        manifest_member = staged.extractfile("./r225_eight_lane_diagnostic_manifest.json")
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read())
        assert manifest["automated_kaggle_actions_present"] is False
        assert manifest["required_label"] == stage_mod.REQUIRED_LABEL


def test_r225_wrapper_keeps_direct_entrypoint_and_has_no_file_magic() -> None:
    source = (ROOT / "submission/r225_eight_lane_diagnostic_main.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree))
    assert "direct.agent(obs_dict)" in source
    assert "return list(action)" in source
    assert "R225_EIGHT_LANE_DIAGNOSTIC_FAILED" in source
