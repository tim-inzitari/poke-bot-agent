from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import types
import zipfile
from pathlib import Path

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


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, body in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 10, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, body)


def _configure_canonical_wheel(
    stage_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    payloads: dict[str, bytes] = {}
    members = []
    wheel_members: dict[str, bytes] = {}
    for index, raw_member in enumerate(stage_mod.CANONICAL_LIBCG_MEMBERS):
        member = dict(raw_member)
        payload = f"canonical-{index}-{member['platform']}".encode()
        payloads[member["package_relative_path"]] = payload
        wheel_members[member["wheel_member"]] = payload
        member["sha256"] = _sha_bytes(payload)
        member["size_bytes"] = len(payload)
        members.append(member)
    wheel = tmp_path / "kaggle_environments-1.32.6-py3-none-any.whl"
    _write_wheel(wheel, wheel_members)
    monkeypatch.setattr(stage_mod, "CANONICAL_LIBCG_MEMBERS", tuple(members))
    monkeypatch.setattr(
        stage_mod, "CANONICAL_LIBCG_WHEEL_SHA256", _sha_bytes(wheel.read_bytes())
    )
    monkeypatch.setattr(stage_mod, "CANONICAL_LIBCG_WHEEL_BYTES", wheel.stat().st_size)
    return wheel, payloads


def _copy_wrapper_sources(destination: Path) -> None:
    for relative in (
        "submission/r228_async_eight_worker_main.py",
        "poke_bot/r228_kaggle_broker.py",
        "poke_bot/r228_kaggle_async_runtime.py",
        "poke_bot/r228_async_shared_tree_queue.py",
        "poke_bot/r225_stock_native_lane.py",
        "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def _clear_staged_package_modules() -> None:
    """Remove only package modules that an exact-stage import owns."""

    for name in tuple(sys.modules):
        if (
            name == "poke_bot"
            or name.startswith("poke_bot.")
            or name == "r228_broker_r195_direct"
        ):
            sys.modules.pop(name, None)


def test_r238_stager_is_deterministic_and_preserves_frozen_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)
    canonical_wheel, canonical_payloads = _configure_canonical_wheel(
        stage_mod, tmp_path, monkeypatch
    )

    files = {
        "main.py": b"def agent(obs):\n    return [0]\n",
        "model.pt": b"r195-model",
        "matchup_tree.json": b"r195-matchup-tree",
        "search_config.json": b'{"frozen":true}\n',
        "cg/libcg.so": b"stock-libcg",
        "cg/api.py": b"API = 'r195'\n",
        "cg/utils.py": b"UTILS = 'r195'\n",
        "cg/sim.py": b"SIM = 'r195'\n",
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
        r195_bundle=archive,
        canonical_libcg_wheel=canonical_wheel,
        output_dir=tmp_path / "one",
        source_root=source_root,
    )
    second = stage_mod.stage_bundle(
        r195_bundle=archive,
        canonical_libcg_wheel=canonical_wheel,
        output_dir=tmp_path / "two",
        source_root=source_root,
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
            "cg/libcg-arm64.so",
            "cg/libcg.dylib",
            "cg/cg.dll",
            "cg/api.py",
            "cg/utils.py",
            "cg/sim.py",
            "poke_bot/r228_kaggle_broker.py",
            "poke_bot/r228_kaggle_async_runtime.py",
            "poke_bot/r228_async_shared_tree_queue.py",
            "poke_bot/r225_stock_native_lane.py",
            "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
            stage_mod.MANIFEST_FILENAME,
        } <= names
        search_config = archive_file.extractfile("./search_config.json")
        assert search_config is not None
        assert search_config.read() == files["search_config.json"]
        for relative, expected in canonical_payloads.items():
            member = archive_file.extractfile("./" + relative)
            assert member is not None
            assert member.read() == expected
        assert canonical_payloads["cg/libcg.so"] != files["cg/libcg.so"]
        for relative in ("cg/api.py", "cg/utils.py", "cg/sim.py"):
            wrapper_member = archive_file.extractfile("./" + relative)
            assert wrapper_member is not None
            assert wrapper_member.read() == files[relative]
        manifest_file = archive_file.extractfile("./" + stage_mod.MANIFEST_FILENAME)
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read())
        assert manifest["entrypoint_sha256"] == first["entrypoint_sha256"]
        assert manifest["schema"] == stage_mod.SCHEMA
        assert manifest["role"] == "isolated_r238_two_lane_bounded_mcts_fallback_diagnostic"
        assert manifest["required_label"] == stage_mod.REQUIRED_LABEL
        assert manifest["complete_action_cap"] == 65_536
        assert manifest["broker_contract"] == stage_mod.broker_contract()
        assert manifest["r240_hybrid_scheduler"] == stage_mod.r240_hybrid_scheduler_contract()
        assert (
            manifest["deterministic_continuation"]
            == stage_mod.deterministic_continuation_contract()
        )
        assert manifest["r240_required_preflight_receipts"] == list(
            stage_mod.R240_REQUIRED_REGRESSION_RECEIPTS
        )
        assert (
            manifest[
                "owner_proven_deterministic_terminal_win_this_turn_revision"
            ]
            == stage_mod.PROVEN_TERMINAL_WIN_REVISION
        )
        assert (
            manifest["r246_proven_deterministic_terminal_win_this_turn"]
            == stage_mod.r246_proven_deterministic_terminal_win_contract()
        )
        assert manifest["r246_required_preflight_receipts"] == [
            stage_mod.PROVEN_TERMINAL_WIN_REGRESSION_RECEIPT
        ]
        assert (
            manifest["r244_handle_scoped_search_identity"]
            == stage_mod.r244_handle_scoped_search_identity_contract()
        )
        assert manifest["r244_required_preflight_receipts"] == [
            stage_mod.R244_HANDLE_SCOPED_SEARCH_IDENTITY_REGRESSION_RECEIPT
        ]
        assert manifest["r225_typed_contract"] == stage_mod.r225_typed_contract_identity(
            source_root / stage_mod.R225_TYPED_CONTRACT_PATH
        )
        assert manifest["lane_count"] == 2
        assert manifest["required_search_lifecycle_counts"] == {
            "search_begin_calls": 2,
            "search_end_calls": 2,
            "search_release_calls": 2,
        }
        assert manifest["phase1_kaggle_resource_bounds"] == {
            "vcpus": 2,
            "ram_gib": 12.2,
            "hdd_gib": 11.8,
            "archive_mib": 197.7,
            "gpu_environment_inferred_from_resource_envelope": False,
            "runtime_cuda_observation_required_before_search": True,
            "archive_max_bytes": stage_mod.PHASE1_ARCHIVE_MAX_BYTES,
        }
        assert manifest["canonical_libcg_contract"] == stage_mod.canonical_libcg_contract()
        assert manifest["exact_frozen_base"] == stage_mod.exact_frozen_base_contract()
        assert manifest["canonical_libcg_contract"]["schema"] == (
            "poke_bot.canonical_libcg_r236/v1"
        )
        assert manifest["canonical_libcg_contract"]["typed_source"] == (
            "state/canonical-libcg-r236.json"
        )
        assert manifest["canonical_libcg_contract"]["upstream_provenance"] == {
            **stage_mod.CANONICAL_LIBCG_UPSTREAM_PROVENANCE,
            "wheel_filename": stage_mod.CANONICAL_LIBCG_WHEEL_FILENAME,
            "wheel_sha256": _sha_bytes(canonical_wheel.read_bytes()),
            "wheel_size_bytes": canonical_wheel.stat().st_size,
        }
        assert manifest["r225_package_preflight"] == {
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
            "all_four_canonical_native_members_checksum_and_size_verified": True,
            "old_or_mixed_native_members_rejected": True,
            "required_native_exports": [
                "AgentStart",
                "BattleStart",
                "SearchBegin",
                "SearchStep",
                "SearchRelease",
                "SearchEnd",
            ],
        }
        assert manifest["canonical_native_member_sha256"] == {
            path: _sha_bytes(payload) for path, payload in canonical_payloads.items()
        }
        assert {
            path: member["size_bytes"]
            for path, member in manifest["canonical_native_members"].items()
        } == {path: len(payload) for path, payload in canonical_payloads.items()}
        assert manifest["frozen_r195_cg_wrapper_members"] == {
            "cg/api.py": _sha_bytes(files["cg/api.py"]),
            "cg/sim.py": _sha_bytes(files["cg/sim.py"]),
            "cg/utils.py": _sha_bytes(files["cg/utils.py"]),
        }
        assert "cg/libcg.so" not in manifest["preserved_members"]
        assert (
            manifest["subprocess_containment_identity"]
            == stage_mod.SUBPROCESS_CONTAINMENT_IDENTITY
        )
        assert manifest["broker_contract"]["subprocess_containment"][
            "bounded_reap_required"
        ]
        r225_contract_member = archive_file.extractfile(
            "./" + stage_mod.R225_TYPED_CONTRACT_PATH
        )
        assert r225_contract_member is not None
        assert _sha_bytes(r225_contract_member.read()) == manifest["r225_typed_contract"][
            "sha256"
        ]

    assert first["complete_action_cap"] == 65_536
    assert first["broker_contract"] == stage_mod.broker_contract()
    assert first["broker_contract"]["action_timeout_seconds"] == 4.0
    assert first["broker_contract"]["search_seconds"] == 2.0
    assert first["broker_contract"]["startup_timeout_seconds"] == 30.0
    assert first["broker_contract"]["reap_grace_seconds"] == 0.25
    assert first["lane_count"] == 2
    assert first["required_search_lifecycle_counts"] == {
        "search_begin_calls": 2,
        "search_end_calls": 2,
        "search_release_calls": 2,
    }
    assert first["canonical_libcg_contract"] == stage_mod.canonical_libcg_contract()
    assert first["canonical_native_member_sha256"] == {
        path: _sha_bytes(payload) for path, payload in canonical_payloads.items()
    }
    assert first["exact_frozen_base"] == stage_mod.exact_frozen_base_contract()
    assert first["r240_hybrid_scheduler"] == stage_mod.r240_hybrid_scheduler_contract()
    assert (
        first["deterministic_continuation"]
        == stage_mod.deterministic_continuation_contract()
    )
    assert first["r240_required_preflight_receipts"] == list(
        stage_mod.R240_REQUIRED_REGRESSION_RECEIPTS
    )
    assert (
        first["owner_proven_deterministic_terminal_win_this_turn_revision"]
        == stage_mod.PROVEN_TERMINAL_WIN_REVISION
    )
    assert (
        first["r246_proven_deterministic_terminal_win_this_turn"]
        == stage_mod.r246_proven_deterministic_terminal_win_contract()
    )
    assert first["r246_required_preflight_receipts"] == [
        stage_mod.PROVEN_TERMINAL_WIN_REGRESSION_RECEIPT
    ]
    assert (
        first["r244_handle_scoped_search_identity"]
        == stage_mod.r244_handle_scoped_search_identity_contract()
    )
    assert first["r244_required_preflight_receipts"] == [
        stage_mod.R244_HANDLE_SCOPED_SEARCH_IDENTITY_REGRESSION_RECEIPT
    ]
    assert first["r225_typed_contract"] == stage_mod.r225_typed_contract_identity(
        source_root / stage_mod.R225_TYPED_CONTRACT_PATH
    )
    assert first["r225_package_preflight"] == stage_mod.canonical_libcg_contract()[
        "r225_package_preflight"
    ]
    assert (
        first["subprocess_containment_identity"]
        == stage_mod.SUBPROCESS_CONTAINMENT_IDENTITY
    )
    assert first["failed_kaggle_validation_evidence"] == {
        "submission_id": 55_416_396,
        "episode_id": 91_766_923,
        "submission_message": "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY",
        "submission_status": "SubmissionStatus.ERROR",
        "episode_terminal_status": "TIMEOUT",
        "final_root_ordered_legal_action_count": 2,
        "final_unreturned_callback_elapsed_seconds": 438.994125,
        "failed_submission_must_be_preserved": True,
    }
    assert first["required_label"] == "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"


def test_r238_exact_staged_broker_uses_frozen_turn_order_resolver_not_workspace_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restaged child must replay IsFirst with frozen r195-only imports.

    The frozen base deliberately lacks the newer workspace
    ``features.forced_go_first_action`` helper.  Importing the actual staged
    broker, its staged r195 direct entrypoint, and its staged feature module
    catches an accidental host-package import before a Kaggle preflight can.
    """

    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)
    canonical_wheel, _ = _configure_canonical_wheel(stage_mod, tmp_path, monkeypatch)

    frozen_direct = (
        ROOT / "output/r195-train-preimage/submission/main.py"
    ).read_bytes()
    files = {
        "main.py": frozen_direct,
        "model.pt": b"r195-model",
        "matchup_tree.json": b"r195-matchup-tree",
        "search_config.json": b'{"frozen":true}\n',
        "cg/libcg.so": b"stock-libcg",
        "cg/api.py": b"API = 'r195'\n",
        "cg/utils.py": b"UTILS = 'r195'\n",
        "cg/sim.py": b"SIM = 'r195'\n",
        "poke_bot/__init__.py": b"# frozen r195 package\n",
        # The compatibility regression is meaningful only when this staged
        # base has no workspace-only helper to fall back to.
        "poke_bot/features.py": (
            b"FROZEN_R195_FEATURES = True\n"
            b"def build_option_tokens(observation, actions):\n"
            b"    return {'frozen': True, 'actions': list(actions)}\n"
        ),
    }
    archive = tmp_path / "r195-turn-order.tar.gz"
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

    staged_receipt = stage_mod.stage_bundle(
        r195_bundle=archive,
        canonical_libcg_wheel=canonical_wheel,
        output_dir=tmp_path / "output",
        source_root=source_root,
    )
    assert staged_receipt["status"] == "staged_not_submitted"
    extracted = tmp_path / "exact-stage"
    with tarfile.open(tmp_path / "output" / stage_mod.ARCHIVE_FILENAME, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")

    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "poke_bot"
        or name.startswith("poke_bot.")
        or name == "r228_broker_r195_direct"
    }
    original_path = list(sys.path)
    original_cwd = Path.cwd()
    try:
        _clear_staged_package_modules()
        sys.path.insert(0, str(extracted))
        staged_broker = importlib.import_module("poke_bot.r228_kaggle_broker")
        staged_features = importlib.import_module("poke_bot.features")
        assert Path(staged_broker.__file__).resolve().is_relative_to(extracted)
        assert Path(staged_features.__file__).resolve().is_relative_to(extracted)
        assert not hasattr(staged_features, "forced_go_first_action")

        direct = staged_broker._child_load_direct(extracted)
        assert Path(direct.__file__).resolve().is_relative_to(extracted)
        observation = {
            "select": {
                "context": "IsFirst",
                "option": [{"type": "No"}, {"type": "Yes"}],
                "minCount": 1,
                "maxCount": 1,
            }
        }
        assert direct._turn_order_choice(observation) == [1]

        class _JournalPolicy:
            def __init__(self) -> None:
                self.board_history: list[object] = []
                self._previous_action_token = None

            def _append_decision_history(self, history_observation: object) -> None:
                self.board_history.append(history_observation)

        class _StagedGameplay:
            def __init__(self, *, policy: object, **_kwargs: object) -> None:
                self.policy = policy

        policy = _JournalPolicy()
        staged_runtime_module = types.ModuleType("poke_bot.r228_kaggle_async_runtime")
        staged_runtime_module.R228AsyncGameplay = _StagedGameplay
        staged_runtime_module.validate_staged_stock_library_identity = (
            lambda _stage: {"member": "cg/libcg.so"}
        )
        monkeypatch.setitem(
            sys.modules,
            "poke_bot.r228_kaggle_async_runtime",
            staged_runtime_module,
        )
        monkeypatch.setattr(
            direct,
            "_ensure_runtime",
            lambda: ([741] * 60, object(), policy),
        )
        monkeypatch.setattr(staged_broker, "_child_load_direct", lambda _stage: direct)
        runtime = staged_broker._child_new_runtime(extracted, object())
        assert runtime.child_frozen_turn_order_choice is direct._turn_order_choice
        staged_broker._child_commit_action(
            runtime,
            {"observation": observation, "action": [1]},
        )
        assert runtime.policy.board_history == []
        assert runtime.policy._previous_action_token is None

        ordinary_observation = {
            "select": {
                "context": "Main",
                "option": [{}, {}],
                "minCount": 1,
                "maxCount": 1,
            }
        }
        staged_broker._child_commit_action(
            runtime,
            {"observation": ordinary_observation, "action": [0]},
        )
        assert runtime.policy.board_history == [ordinary_observation]
        assert runtime.policy._previous_action_token == {
            "frozen": True,
            "actions": [[0]],
        }
    finally:
        _clear_staged_package_modules()
        sys.modules.update(original_modules)
        sys.path[:] = original_path
        os.chdir(original_cwd)


def test_r238_stager_rejects_a_wheel_with_the_wrong_digest(tmp_path: Path) -> None:
    stage_mod = _load_stage_module()
    wheel = tmp_path / "kaggle_environments-1.32.6-py3-none-any.whl"
    _write_wheel(
        wheel,
        {"kaggle_environments/envs/cabt/cg/libcg.so": b"not-canonical"},
    )

    with pytest.raises(stage_mod.R228StageError, match="wheel digest mismatch"):
        stage_mod.verify_canonical_libcg_wheel(wheel)


def test_r238_stager_cli_requires_canonical_libcg_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_mod = _load_stage_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_r228_async_eight_worker_kaggle_viability.py",
            "--r195-bundle",
            "r195.tar.gz",
            "--output-dir",
            "out",
        ],
    )

    with pytest.raises(SystemExit) as error:
        stage_mod.main()
    assert error.value.code == 2


def test_r238_stager_rejects_a_stale_gpu_resource_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_mod = _load_stage_module()
    monkeypatch.setattr(
        stage_mod,
        "PHASE1_KAGGLE_RESOURCE_BOUNDS",
        {**stage_mod.PHASE1_KAGGLE_RESOURCE_BOUNDS, "gpu_available": False},
    )

    with pytest.raises(stage_mod.R228StageError, match="stale gpu_available"):
        stage_mod.phase1_kaggle_resource_bounds()


def test_r238_stager_rejects_old_or_mixed_native_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)
    canonical_wheel, _ = _configure_canonical_wheel(stage_mod, tmp_path, monkeypatch)
    files = {
        "main.py": b"def agent(obs):\n    return [0]\n",
        "model.pt": b"r195-model",
        "matchup_tree.json": b"r195-matchup-tree",
        "search_config.json": b'{"frozen":true}\n',
        "cg/libcg.so": b"stock-libcg",
        "cg/api.py": b"API = 'r195'\n",
        "cg/libcg-old.so": b"old-native-must-not-survive",
    }
    archive = tmp_path / "r195-mixed.tar.gz"
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
    monkeypatch.setattr(stage_mod, "validate_async_action_authority", lambda *_: None)

    output_dir = tmp_path / "mixed-output"
    with pytest.raises(stage_mod.R228StageError, match="old, missing, or mixed native"):
        stage_mod.stage_bundle(
            r195_bundle=archive,
            canonical_libcg_wheel=canonical_wheel,
            output_dir=output_dir,
            source_root=source_root,
        )
    assert not (output_dir / stage_mod.ARCHIVE_FILENAME).exists()
    assert not (output_dir / stage_mod.RECEIPT_FILENAME).exists()


def test_r238_stager_rejects_legacy_raw_search_id_authority(
    tmp_path: Path,
) -> None:
    """The replacement must prove handle-scoped composites, never raw IDs."""

    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)
    wrapper = source_root / "submission/r228_async_eight_worker_main.py"
    runtime = source_root / "poke_bot/r228_kaggle_async_runtime.py"
    broker = source_root / "poke_bot/r228_kaggle_broker.py"
    queue = source_root / "poke_bot/r228_async_shared_tree_queue.py"
    source = runtime.read_text(encoding="utf-8")
    current = 'receipt, "distinct_search_begin_composite_count"'
    legacy = 'receipt, "distinct_search_begin_id_count"'
    assert current in source
    runtime.write_text(source.replace(current, legacy, 1), encoding="utf-8")

    with pytest.raises(stage_mod.R228StageError, match="r244 handle-scoped"):
        stage_mod.validate_async_action_authority(wrapper, runtime, broker, queue)


def test_r238_stager_requires_pre_search_cuda_runtime_observation(
    tmp_path: Path,
) -> None:
    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)
    wrapper = source_root / "submission/r228_async_eight_worker_main.py"
    runtime = source_root / "poke_bot/r228_kaggle_async_runtime.py"
    broker = source_root / "poke_bot/r228_kaggle_broker.py"
    queue = source_root / "poke_bot/r228_async_shared_tree_queue.py"
    source = broker.read_text(encoding="utf-8")
    current = "capture_cuda_runtime_before_search(model)"
    assert current in source
    broker.write_text(
        source.replace(current, "missing_cuda_runtime_observation(model)", 1),
        encoding="utf-8",
    )

    with pytest.raises(stage_mod.R228StageError, match="pre-search CUDA runtime observation"):
        stage_mod.validate_async_action_authority(wrapper, runtime, broker, queue)


@pytest.mark.parametrize(
    ("relative", "needle", "replacement", "error"),
    (
        (
            "submission/r228_async_eight_worker_main.py",
            '"exact_deterministic_simulator_terminal_win_this_turn"',
            '"not_a_terminal_win_proof"',
            "main.py lacks r246 deterministic terminal-win parent validation",
        ),
        (
            "poke_bot/r228_kaggle_async_runtime.py",
            "    PROVEN_TERMINAL_WIN_PROOF_KIND,\n",
            "    MISSING_TERMINAL_WIN_PROOF_KIND,\n",
            "contained child runtime lacks r246 deterministic terminal-win validation",
        ),
        (
            "poke_bot/r228_kaggle_broker.py",
            '"exact_deterministic_simulator_terminal_win_this_turn"',
            '"not_a_terminal_win_proof"',
            "r238 broker lacks r246 deterministic terminal-win validation",
        ),
        (
            "poke_bot/r228_async_shared_tree_queue.py",
            '"exact_deterministic_simulator_terminal_win_this_turn"',
            '"not_a_terminal_win_proof"',
            "contained child queue lacks r246 deterministic terminal-win proof authority",
        ),
        (
            "poke_bot/r228_async_shared_tree_queue.py",
            '"discovering_lane_id": context.lane_id',
            '"missing_discovering_lane_id": context.lane_id',
            "contained child queue lacks r246 deterministic terminal-win proof authority",
        ),
    ),
)
def test_r238_stager_rejects_incomplete_r246_terminal_win_authority(
    tmp_path: Path,
    relative: str,
    needle: str,
    replacement: str,
    error: str,
) -> None:
    """Every staged authority boundary must preserve the exact r246 proof gate."""

    stage_mod = _load_stage_module()
    source_root = tmp_path / "source"
    _copy_wrapper_sources(source_root)
    target = source_root / relative
    source = target.read_text(encoding="utf-8")
    assert needle in source
    target.write_text(source.replace(needle, replacement, 1), encoding="utf-8")

    with pytest.raises(stage_mod.R228StageError, match=error):
        stage_mod.validate_async_action_authority(
            source_root / "submission/r228_async_eight_worker_main.py",
            source_root / "poke_bot/r228_kaggle_async_runtime.py",
            source_root / "poke_bot/r228_kaggle_broker.py",
            source_root / "poke_bot/r228_async_shared_tree_queue.py",
        )


def _valid_wrapper_source() -> str:
    return """\
from poke_bot.r228_kaggle_broker import COMPLETE_ACTION_CAP, IsolatedR228SearchBroker

SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
R228_ASYNC_SELECTED_ACTION_AUTHORITY = "receipt.selected_action"
DECISION_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_DECISION"
FULL_GAMEPLAY_SUCCESS_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_FULL_GAMEPLAY_SUCCESS"
HARD_FAILURE_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_HARD_FAILURE"
R234_BROKER_ACTION_TIMEOUT_SECONDS = 4.0
R234_BROKER_SEARCH_SECONDS = 2.0
R234_BROKER_STARTUP_TIMEOUT_SECONDS = 30.0
R234_BROKER_REAP_GRACE_SECONDS = 0.25
R238_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED = 3
R238_MAXIMUM_BACKUPS_PER_DECISION = 32
R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD = 0.80
R240_MAX_PRINCIPAL_VARIATION_DEPTH = 8
DEGRADED_MARKER = "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED"
_BROKER = None

def _agent_dir():
    return None

def _direct():
    return direct

def _broker():
    global _BROKER
    if _BROKER is None:
        _BROKER = IsolatedR228SearchBroker(
            stage=_agent_dir(),
            action_timeout_seconds=R234_BROKER_ACTION_TIMEOUT_SECONDS,
            search_seconds=R234_BROKER_SEARCH_SECONDS,
            startup_timeout_seconds=R234_BROKER_STARTUP_TIMEOUT_SECONDS,
            reap_grace_seconds=R234_BROKER_REAP_GRACE_SECONDS,
        )
    return _BROKER

def agent(obs):
    direct_action = list(_direct().agent(obs))
    selected, receipt, fault = _broker().select(obs, direct_action)
    return selected
"""


def _valid_runtime_source() -> str:
    return """\
import hashlib
import json

SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
DECISION_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_DECISION"
R228_SIMULATOR_LANE_COUNT = 2
R238_DEFAULT_SEARCH_SECONDS = 2.0
R238_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R238_STABLE_ROOT_LEADER_OBSERVATIONS = 3
R238_MAXIMUM_BACKUPS_PER_DECISION = 32

def canonical_observation_fingerprint(raw):
    canonical = json.dumps(
        dict(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()

def _validate_principal_variation(receipt):
    return receipt.principal_variation

def select():
    engine = PersistentAsyncSharedTreeMCTS(
        lane_count=R228_SIMULATOR_LANE_COUNT,
        minimum_backups_before_stability=R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
        stable_root_leader_observations=R238_STABLE_ROOT_LEADER_OBSERVATIONS,
        maximum_backups_per_decision=R238_MAXIMUM_BACKUPS_PER_DECISION,
    )
    receipt = engine.run_decision(deadline_monotonic=deadline)
    selected = receipt.selected_action
    payload = {
        "stop_reason": receipt.stop_reason,
        "minimum_backups_before_stability": R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
        "stable_root_leader_observations_required": R238_STABLE_ROOT_LEADER_OBSERVATIONS,
        "maximum_backups_per_decision": R238_MAXIMUM_BACKUPS_PER_DECISION,
        "observed_stable_root_leader_observations": receipt.leader_stability_count,
        "root_seat": receipt.root_seat,
        "principal_variation": _validate_principal_variation(receipt),
    }
    return list(selected)
"""


def _valid_broker_source() -> str:
    return """\
import subprocess

SCHEMA = "poke_bot.r228_kaggle_subprocess_broker/v1"
COMPLETE_ACTION_CAP = 65536

def complete(obs):
    return enumerate_action_combos(obs, max_combos=COMPLETE_ACTION_CAP)

class IsolatedR228SearchBroker:
    def __init__(
        self,
        stage,
        action_timeout_seconds,
        search_seconds,
        startup_timeout_seconds,
        reap_grace_seconds,
    ):
        self.action_timeout_seconds = _positive_seconds(
            action_timeout_seconds, fallback=4.0
        )
        self.search_seconds = _positive_seconds(search_seconds, fallback=2.0)
        self.startup_timeout_seconds = _positive_seconds(
            startup_timeout_seconds, fallback=30.0
        )
        self.reap_grace_seconds = _positive_seconds(
            reap_grace_seconds, fallback=0.25
        )

    def begin_game(self):
        return None

    def note_direct_action(self, obs, actual_action):
        return None

    def select(self, obs, direct_action):
        return direct_action, {"selected_action": direct_action}, None

    def close(self):
        return None

    def _dispose_child(self):
        child = subprocess.Popen(["broker"])
        child.terminate()
        child.wait(timeout=self.reap_grace_seconds)
        child.kill()
"""


def _write_authority_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    wrapper = tmp_path / "main.py"
    runtime = tmp_path / "runtime.py"
    broker = tmp_path / "broker.py"
    wrapper.write_text(_valid_wrapper_source(), encoding="utf-8")
    runtime.write_text(_valid_runtime_source(), encoding="utf-8")
    broker.write_text(_valid_broker_source(), encoding="utf-8")
    return wrapper, runtime, broker


def test_r238_stage_rejects_a_direct_policy_side_probe(tmp_path: Path) -> None:
    stage_mod = _load_stage_module()
    wrapper, runtime, broker = _write_authority_sources(tmp_path)
    runtime.write_text(
        _valid_runtime_source().replace(
            "    return list(selected)\n",
            "    return [0]\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(stage_mod.R228StageError, match="selected action"):
        stage_mod.validate_async_action_authority(wrapper, runtime, broker)


def test_r238_stage_rejects_a_child_that_is_not_exactly_two_lane(
    tmp_path: Path,
) -> None:
    stage_mod = _load_stage_module()
    wrapper, runtime, broker = _write_authority_sources(tmp_path)
    runtime.write_text(
        _valid_runtime_source().replace(
            "R228_SIMULATOR_LANE_COUNT = 2",
            "R228_SIMULATOR_LANE_COUNT = 8",
        ),
        encoding="utf-8",
    )

    with pytest.raises(stage_mod.R228StageError, match="exactly two lanes"):
        stage_mod.validate_async_action_authority(wrapper, runtime, broker)


def test_r238_stage_rejects_main_without_retained_direct_broker_fallback(
    tmp_path: Path,
) -> None:
    stage_mod = _load_stage_module()
    wrapper, runtime, broker = _write_authority_sources(tmp_path)
    wrapper.write_text(
        _valid_wrapper_source().replace(
            "    selected, receipt, fault = _broker().select(obs, direct_action)\n"
            "    return selected\n",
            "    return _broker().select(obs, [0])[0]\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(stage_mod.R228StageError, match="precompute and retain direct fallback"):
        stage_mod.validate_async_action_authority(wrapper, runtime, broker)
