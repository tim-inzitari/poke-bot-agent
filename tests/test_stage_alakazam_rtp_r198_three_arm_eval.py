"""Focused safety coverage for the isolated r198 three-arm stage surface."""

from __future__ import annotations

import ast
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts" / "stage_alakazam_rtp_r198_three_arm_eval.py"
SNAPSHOT = ROOT / "scripts" / "stage_alakazam_rtp_r198_three_arm_eval_source_snapshot.py"
UNIT = ROOT / "deploy" / "systemd" / "pokebot-alakazam-rtp-r198-three-arm-eval.service"


def _r210_guard_output_root(tmp_path: Path) -> Path:
    """Use a lexical workspace path; macOS tmp roots may traverse `/var`."""

    return ROOT / ".r198-r210-stage-guard-tests" / tmp_path.name


@pytest.fixture(scope="module")
def stage_module():
    spec = importlib.util.spec_from_file_location("stage_r198_three_arm_test", STAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_stage_uses_only_the_direct_v2_materialize_run_compile_chain() -> None:
    source = STAGE.read_text(encoding="utf-8")

    assert "materialize_r198_evaluation_inputs" in source
    assert "poke_bot.rtp_three_arm_evaluation_runner" in source
    assert "run_three_arm_evaluation" in source
    assert "compile_three_arm_receipt" in source
    assert "_materialize_cohort" not in source
    assert "_cohort_plan" not in source
    assert "poke_bot.rtp_three_arm_runner" not in source
    assert "R198_MAX_NEURAL_PASSES = 256" in source
    assert "R198_MAX_ACTION_COMBOS = 1024" in source
    assert "R198_NORMAL_PASSES = 6" in source
    assert "R198_FORCED_REPLAN_PASSES = 5" in source


@pytest.mark.unit
def test_stage_pins_the_canonical_capability_and_final_eval_cg_closure() -> None:
    source = STAGE.read_text(encoding="utf-8")
    snapshot = SNAPSHOT.read_text(encoding="utf-8")

    assert "rtp-pairing-v2-probes-canonical-seal-v2" in source
    assert "true-rng-pairing-capability-v2.json" in source
    assert "46ad92e5927aa254728769e184e57840fa1b5b16c2ecd7a5f6da91755cfdf381" in source
    assert "PAIRING_CAPABILITY_BYTES = 3207" in source
    assert '"engine_artifact": 0o444' in source
    assert '"runtime_library": inputs["evaluation_cg"]["library"]' in source
    assert 'prepared_closure.get("runtime_library")' in source
    for text in (source, snapshot):
        assert "419ad46a9b31b9fdc040b851b553108b1bd038b68acadccb4dc9c38bfd35bbe0" in text
        assert "cbdffe7fe99c9c29d83cc6dd3530b1c406ce7f4d0f99920ca6fc45624e0e25a7" in text
        assert "distinct_dso_handles" in text
        assert "RtpPairingSnapshotInitialize" in text


@pytest.mark.unit
def test_stage_pins_snapshot_local_inputs_and_fresh_attempt10_output(
    stage_module,
) -> None:
    assert stage_module.R197_COMPLETION_RECEIPT_SHA256 == (
        "sha256:b0c209257ed401bf9c5fe5a1ee17be1d1cdc01a1f9780e3e0d23ce8fa5f80737"
    )
    assert stage_module.R197_COMPLETION_RECEIPT_BYTES == 113366
    assert stage_module.MATCHUP_ADAPTER_ROSTER_SHA256 == (
        "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
    )
    assert stage_module.MATCHUP_ADAPTER_ROSTER_BYTES == 11899
    assert stage_module.MATCHUP_ADAPTER_ROSTER_MODE == 0o444
    assert str(stage_module.DEFAULT_OUTPUT_ROOT) == (
        "/home/pokebot/poke-bot-agent/outputs/rtp_fleet/"
        "alakazam-r198-three-arm-eval-attempt10"
    )
    for prior_attempt in range(1, 10):
        assert stage_module.DEFAULT_OUTPUT_ROOT != Path(
            "/home/pokebot/poke-bot-agent/outputs/rtp_fleet/"
            f"alakazam-r198-three-arm-eval-attempt{prior_attempt}"
        )
    for diagnostic_root in (
        "alakazam-r198-three-arm-eval-attempt5-first-worker-diagnostic-28562f86",
        "alakazam-r198-three-arm-eval-attempt6-direct-worker-diagnostic-c3fa04d3",
    ):
        assert stage_module.DEFAULT_OUTPUT_ROOT != Path(
            "/home/pokebot/poke-bot-agent/outputs/rtp_fleet/" + diagnostic_root
        )


@pytest.mark.unit
def test_r210_abandonment_rejects_run_before_any_output_root_access(
    stage_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = _r210_guard_output_root(tmp_path)
    calls: list[tuple[Path, bool]] = []
    original_output_root = stage_module._output_root

    def track_output_root(path: Path, *, create: bool) -> Path:
        calls.append((path, create))
        return original_output_root(path, create=create)

    def must_not_preflight(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("r210-abandoned --run must not reach evaluator preflight")

    monkeypatch.setattr(stage_module, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(stage_module, "_output_root", track_output_root)
    monkeypatch.setattr(stage_module, "_preflight", must_not_preflight)

    assert stage_module.main(["--run", "--output-root", str(output_root)]) == 2

    captured = capsys.readouterr()
    assert "legacy recursive RTP was abandoned by revision 210" in captured.err
    assert calls == []
    assert not output_root.exists()


@pytest.mark.unit
def test_r210_guard_fails_closed_on_a_malformed_contract_before_output_write(
    stage_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = _r210_guard_output_root(tmp_path)
    malformed = tmp_path.resolve() / "alakazam-rtp-abandonment-r210.json"
    malformed.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(stage_module, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(stage_module, "R210_ABANDONMENT_CONTRACT_PATH", malformed)
    monkeypatch.setattr(
        stage_module,
        "_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("malformed r210 contract must not reach evaluator preflight")
        ),
    )

    assert stage_module.main(["--run", "--output-root", str(output_root)]) == 2

    captured = capsys.readouterr()
    assert "revision-210 legacy RTP abandonment contract is incomplete" in captured.err
    assert not output_root.exists()


@pytest.mark.unit
def test_r210_keeps_check_read_only_and_independent_of_the_run_guard(
    stage_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = _r210_guard_output_root(tmp_path)
    observed: list[bool] = []

    def fake_preflight(args, *, require_snapshot: bool):  # type: ignore[no-untyped-def]
        observed.append(require_snapshot)
        return {"fixture": True}

    def must_not_read_r210() -> dict:
        raise AssertionError("--check must not consult the r210 run-only guard")

    monkeypatch.setattr(stage_module, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(stage_module, "_preflight", fake_preflight)
    monkeypatch.setattr(stage_module, "_r210_abandonment_contract", must_not_read_r210)
    monkeypatch.setattr(
        stage_module,
        "_base_spec_payload",
        lambda inputs: ({}, {"fixture_base": True}),
    )
    monkeypatch.setattr(stage_module, "_load_input_materializer", lambda: object())

    assert stage_module.main(["--check", "--output-root", str(output_root)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert observed == [True]
    assert payload["status"] == "preflight_complete_no_writes"
    assert not output_root.exists()


@pytest.mark.unit
@pytest.mark.parametrize("value", (0, 0.0))
def test_stage_accepts_only_real_numeric_zero_gate_weights(stage_module, value) -> None:
    assert stage_module._is_zero_gate_weight(value) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", (None, False, True, "0", [], -1, 0.25))
def test_stage_rejects_coercible_or_nonzero_gate_weights(stage_module, value) -> None:
    assert stage_module._is_zero_gate_weight(value) is False


@pytest.mark.unit
def test_stage_zero_overlap_counts_are_strict_json_integers(stage_module) -> None:
    assert stage_module._is_zero_count(0) is True
    for value in (None, False, True, "0", 0.0, -1, 1):
        assert stage_module._is_zero_count(value) is False


@pytest.mark.unit
def test_stage_compares_identity_core_across_verified_mode_wrappers(stage_module) -> None:
    stage_identity = {
        "path": "/immutable/capability.json",
        "sha256": "sha256:" + "a" * 64,
        "bytes": 3207,
    }
    factory_identity = {**stage_identity, "mode": 0o444}

    assert stage_module._same_file_identity(factory_identity, stage_identity) is True
    assert stage_module._same_file_identity(
        {**factory_identity, "bytes": 3208}, stage_identity
    ) is False
    assert stage_module._same_file_identity(
        {key: value for key, value in factory_identity.items() if key != "path"},
        stage_identity,
    ) is False


@pytest.mark.unit
def test_stage_uses_strict_zero_guards_for_registry_and_source_proof() -> None:
    source = STAGE.read_text(encoding="utf-8")

    assert "not _is_zero_gate_weight(row.get(\"gate_weight\"))" in source
    assert "not _is_zero_count(proof.get(\"source_identity_overlap_count\"))" in source
    assert "not _is_zero_count(computation.get(\"intersection_episode_count\"))" in source


def _metadata_parity_fixture(stage_module, tmp_path: Path) -> tuple[dict, dict]:
    engine = tmp_path / "libcg-pairing.so"
    public = tmp_path / "libcg-public.so"
    engine.write_bytes(b"pairing-engine")
    public.write_bytes(b"public-engine")
    engine.chmod(0o444)
    public.chmod(0o444)
    engine_identity = stage_module._file_identity(engine, label="pairing engine")
    public_identity = stage_module._file_identity(public, label="public engine")
    metadata = {
        "schema": (
            "poke_bot.recursive_turn_planner."
            "true_rng_pairing_eval_cg_metadata_parity/v1"
        ),
        "status": "passed",
        "independent_processes": True,
        "pairing_engine": engine_identity,
        "public_cg_engine": public_identity,
        "all_card_canonical_sha256": "sha256:" + "1" * 64,
        "all_attack_canonical_sha256": "sha256:" + "2" * 64,
        "public_all_card_raw_sha256": "sha256:" + "3" * 64,
        "pairing_all_card_raw_sha256": "sha256:" + "3" * 64,
        "public_all_attack_raw_sha256": "sha256:" + "4" * 64,
        "pairing_all_attack_raw_sha256": "sha256:" + "4" * 64,
        "public_initialized_before_pairing": True,
        "pairing_private_initialize_after_public_passed": True,
        "distinct_dso_handles": True,
    }
    metadata_path = tmp_path / "metadata-parity.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_path.chmod(0o444)
    return (
        stage_module._file_identity(metadata_path, label="metadata parity"),
        engine_identity,
    )


@pytest.mark.unit
def test_stage_reads_dual_dso_proof_from_bound_metadata_artifact(
    stage_module, tmp_path: Path
) -> None:
    metadata_identity, engine_identity = _metadata_parity_fixture(stage_module, tmp_path)

    binding = stage_module._eval_cg_metadata_parity_binding(
        metadata_identity, engine_identity
    )

    assert binding["identity"] == metadata_identity
    assert binding["pairing_engine"] == engine_identity


@pytest.mark.unit
def test_stage_rejects_incomplete_bound_metadata_artifact(
    stage_module, tmp_path: Path
) -> None:
    metadata_identity, engine_identity = _metadata_parity_fixture(stage_module, tmp_path)
    metadata_path = Path(metadata_identity["path"])
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["distinct_dso_handles"] = False
    metadata_path.chmod(0o644)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    metadata_path.chmod(0o444)
    changed_identity = stage_module._file_identity(metadata_path, label="metadata parity")

    with pytest.raises(stage_module.StageError, match="distinct_dso_handles"):
        stage_module._eval_cg_metadata_parity_binding(
            changed_identity, engine_identity
        )


@pytest.mark.unit
def test_stage_registers_materializer_module_during_dataclass_execution(
    stage_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror the production dynamic loader that defines dataclasses on import."""

    module_name = stage_module.INPUT_MATERIALIZER_MODULE_NAME
    prior = sys.modules.pop(module_name, None)
    loaded = None
    try:
        loaded = stage_module._load_input_materializer()
        assert sys.modules.get(module_name) is loaded
        assert callable(loaded.materialize_r198_evaluation_inputs)
        assert loaded.PAIRED_CELL_COUNT == 1000
    finally:
        if loaded is not None and sys.modules.get(module_name) is loaded:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_removes_partial_materializer_module_after_execution_failure(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_name = stage_module.INPUT_MATERIALIZER_MODULE_NAME
    prior = sys.modules.pop(module_name, None)
    broken = tmp_path / "broken_materializer.py"
    broken.write_text("raise SystemExit('intentional loader failure')\n", encoding="utf-8")
    monkeypatch.setattr(stage_module, "INPUT_MATERIALIZER_RELATIVE", broken)
    try:
        with pytest.raises(stage_module.StageError, match="cannot execute"):
            stage_module._load_input_materializer()
        assert module_name not in sys.modules
    finally:
        if sys.modules.get(module_name) is not None:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_rejects_an_occupied_materializer_module_slot(stage_module) -> None:
    module_name = stage_module.INPUT_MATERIALIZER_MODULE_NAME
    prior = sys.modules.pop(module_name, None)
    sentinel = object()
    sys.modules[module_name] = sentinel  # type: ignore[assignment]
    try:
        with pytest.raises(stage_module.StageError, match="already occupied"):
            stage_module._load_input_materializer()
        assert sys.modules[module_name] is sentinel
    finally:
        if sys.modules.get(module_name) is sentinel:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


def _r197_validator_loader_inputs(
    stage_module, tmp_path: Path, source: str
) -> tuple[Path, dict[str, list[dict[str, object]]], Path]:
    root = tmp_path / "r197-validator-loader"
    helper = root / "scripts" / "stage_alakazam_rtp_r197_source_snapshot.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(source, encoding="utf-8")
    helper.chmod(stage_module.R197_SOURCE_VALIDATOR_MODE)
    return (
        root,
        {
            "source_entries": [
                {
                    "path": "scripts/stage_alakazam_rtp_r197_source_snapshot.py",
                    "type": "file",
                    "mode": stage_module.R197_SOURCE_VALIDATOR_MODE,
                    "size": helper.stat().st_size,
                    "sha256": stage_module._sha256(helper),
                }
            ]
        },
        helper,
    )


def _r197_validator_module_name(stage_module, helper: Path) -> str:
    return "r197_source_snapshot_validator_for_r198_" + stage_module._sha256(helper)[7:19]


@pytest.mark.unit
def test_stage_registers_r197_validator_during_dataclass_execution(
    stage_module, tmp_path: Path
) -> None:
    root, manifest, helper = _r197_validator_loader_inputs(
        stage_module,
        tmp_path,
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class _Proof:\n"
        "    value: str\n"
        "def validate_published_root(root):\n"
        "    return {'proof': _Proof(str(root)).value}\n",
    )
    module_name = _r197_validator_module_name(stage_module, helper)
    prior = sys.modules.pop(module_name, None)
    loaded = None
    try:
        validator, _ = stage_module._load_snapshot_validator(root, manifest)
        loaded = sys.modules[module_name]
        assert validator(root) == {"proof": str(root)}
        assert loaded.__name__ == module_name
    finally:
        if loaded is not None and sys.modules.get(module_name) is loaded:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_rejects_an_occupied_r197_validator_module_slot(
    stage_module, tmp_path: Path
) -> None:
    root, manifest, helper = _r197_validator_loader_inputs(
        stage_module, tmp_path, "def validate_published_root(root):\n    return {}\n"
    )
    module_name = _r197_validator_module_name(stage_module, helper)
    prior = sys.modules.pop(module_name, None)
    sentinel = object()
    sys.modules[module_name] = sentinel  # type: ignore[assignment]
    try:
        with pytest.raises(stage_module.StageError, match="already occupied"):
            stage_module._load_snapshot_validator(root, manifest)
        assert sys.modules[module_name] is sentinel
    finally:
        if sys.modules.get(module_name) is sentinel:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_removes_partial_r197_validator_after_base_exception(
    stage_module, tmp_path: Path
) -> None:
    root, manifest, helper = _r197_validator_loader_inputs(
        stage_module, tmp_path, "raise SystemExit('intentional loader failure')\n"
    )
    module_name = _r197_validator_module_name(stage_module, helper)
    prior = sys.modules.pop(module_name, None)
    try:
        with pytest.raises(stage_module.StageError, match="cannot execute pinned r197"):
            stage_module._load_snapshot_validator(root, manifest)
        assert module_name not in sys.modules
    finally:
        if sys.modules.get(module_name) is not None:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


def _r198_snapshot_validator_helper(tmp_path: Path, source: str) -> Path:
    helper = tmp_path / "stage_alakazam_rtp_r198_three_arm_eval_source_snapshot.py"
    helper.write_text(source, encoding="utf-8")
    return helper


@pytest.mark.unit
def test_stage_registers_r198_snapshot_validator_during_dataclass_execution(
    stage_module, tmp_path: Path
) -> None:
    helper = _r198_snapshot_validator_helper(
        tmp_path,
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class _Proof:\n"
        "    value: str\n"
        "def validate_published_root(root):\n"
        "    return {'proof': _Proof(str(root)).value}\n",
    )
    module_name = stage_module.SOURCE_SNAPSHOT_VALIDATOR_MODULE_NAME
    prior = sys.modules.pop(module_name, None)
    loaded = None
    try:
        validator = stage_module._load_eval_source_snapshot_validator(helper)
        loaded = sys.modules[module_name]
        assert validator(tmp_path) == {"proof": str(tmp_path)}
        assert loaded.__name__ == module_name
    finally:
        if loaded is not None and sys.modules.get(module_name) is loaded:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_rejects_an_occupied_r198_snapshot_validator_module_slot(
    stage_module, tmp_path: Path
) -> None:
    helper = _r198_snapshot_validator_helper(
        tmp_path, "def validate_published_root(root):\n    return {}\n"
    )
    module_name = stage_module.SOURCE_SNAPSHOT_VALIDATOR_MODULE_NAME
    prior = sys.modules.pop(module_name, None)
    sentinel = object()
    sys.modules[module_name] = sentinel  # type: ignore[assignment]
    try:
        with pytest.raises(stage_module.StageError, match="already occupied"):
            stage_module._load_eval_source_snapshot_validator(helper)
        assert sys.modules[module_name] is sentinel
    finally:
        if sys.modules.get(module_name) is sentinel:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_removes_partial_r198_snapshot_validator_after_base_exception(
    stage_module, tmp_path: Path
) -> None:
    helper = _r198_snapshot_validator_helper(
        tmp_path, "raise SystemExit('intentional loader failure')\n"
    )
    module_name = stage_module.SOURCE_SNAPSHOT_VALIDATOR_MODULE_NAME
    prior = sys.modules.pop(module_name, None)
    try:
        with pytest.raises(stage_module.StageError, match="cannot execute r198 evaluation source"):
            stage_module._load_eval_source_snapshot_validator(helper)
        assert module_name not in sys.modules
    finally:
        if sys.modules.get(module_name) is not None:
            del sys.modules[module_name]
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.unit
def test_stage_loads_materializer_once_per_check_or_run_path() -> None:
    tree = ast.parse(STAGE.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    assert len(calls(functions["_materialize_evaluation_inputs"], "_load_input_materializer")) == 1
    assert len(calls(functions["_run"], "_load_input_materializer")) == 0
    assert len(calls(functions["_run"], "_materialize_evaluation_inputs")) == 1
    assert len(calls(functions["main"], "_load_input_materializer")) == 1
    assert len(
        [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_load_input_materializer"
        ]
    ) == 2


def _write_readonly_identity(stage_module, path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return stage_module._file_identity(path, label=path.name)


@pytest.mark.unit
def test_stage_binds_exact_snapshot_local_matchup_adapter_roster(
    stage_module, tmp_path: Path
) -> None:
    snapshot_root = tmp_path / "source-snapshot"
    roster_path = snapshot_root / stage_module.MATCHUP_ADAPTER_ROSTER_RELATIVE
    roster_bytes = (
        ROOT / stage_module.MATCHUP_ADAPTER_ROSTER_RELATIVE
    ).read_bytes()
    _write_readonly_identity(stage_module, roster_path, roster_bytes)
    manifest_path = snapshot_root / stage_module.SOURCE_SNAPSHOT_MANIFEST_NAME
    manifest_payload = {
        "schema": stage_module.SOURCE_SNAPSHOT_SCHEMA,
        "source_entries": [
            {
                "path": stage_module.MATCHUP_ADAPTER_ROSTER_RELATIVE.as_posix(),
                "type": "file",
                "mode": 0o444,
                "size": len(roster_bytes),
                "sha256": stage_module._sha256(roster_path),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest_path.chmod(0o444)
    evaluation_snapshot = {
        "root": str(snapshot_root),
        "manifest": stage_module._file_identity(
            manifest_path, label="source snapshot manifest"
        ),
    }

    binding = stage_module._matchup_adapter_roster_binding(evaluation_snapshot)

    assert binding["identity"]["path"] == str(roster_path)
    assert binding["identity"]["sha256"] == stage_module.MATCHUP_ADAPTER_ROSTER_SHA256
    assert binding["identity"]["bytes"] == stage_module.MATCHUP_ADAPTER_ROSTER_BYTES
    assert stat.S_IMODE(roster_path.lstat().st_mode) == 0o444
    assert binding["verification_status"] == "valid"

    roster_path.chmod(0o644)
    roster_path.write_bytes(roster_bytes + b"\n")
    roster_path.chmod(0o444)
    with pytest.raises(stage_module.StageError, match="checksum"):
        stage_module._matchup_adapter_roster_binding(evaluation_snapshot)


@pytest.mark.unit
def test_candidate_snapshot_binds_immutable_completion_copy_to_live_identity(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path / "source-snapshot"
    package_root = snapshot_root / "evaluation-artifacts" / "r197-candidate"
    package_root.mkdir(parents=True)
    artifact_paths = {
        "parent_checkpoint": package_root / "parent-checkpoint.pt",
        "sidecar": package_root / "rtp-shadow-planner.pt",
        "sidecar_receipt": package_root / "rtp-shadow-planner.receipt.json",
        "completion_receipt": package_root / "r197-completion-receipt.json",
        "deck": package_root / "deck.csv",
        "matchup_tree": package_root / "matchup-tree.json",
    }
    artifacts = {
        key: _write_readonly_identity(stage_module, path, key.encode("utf-8"))
        for key, path in artifact_paths.items()
    }
    live_receipt = tmp_path / "live-candidate" / "r197-receipt.json"
    live_receipt.parent.mkdir()
    live_receipt.write_bytes(b"completion_receipt")
    live_receipt.chmod(0o664)
    live_identity = stage_module._file_identity(live_receipt, label="live completion")
    candidate_contract = tmp_path / "live-candidate" / "candidate-contract.json"

    monkeypatch.setattr(
        stage_module, "PARENT_SHA256", artifacts["parent_checkpoint"]["sha256"]
    )
    monkeypatch.setattr(stage_module, "SIDECAR_SHA256", artifacts["sidecar"]["sha256"])
    monkeypatch.setattr(
        stage_module,
        "R197_COMPLETION_RECEIPT_SHA256",
        artifacts["completion_receipt"]["sha256"],
    )
    monkeypatch.setattr(
        stage_module,
        "R197_COMPLETION_RECEIPT_BYTES",
        artifacts["completion_receipt"]["bytes"],
    )
    monkeypatch.setattr(
        stage_module, "R195_DECK_CSV_SHA256", artifacts["deck"]["sha256"]
    )
    monkeypatch.setattr(
        stage_module,
        "R195_MATCHUP_TREE_SHA256",
        artifacts["matchup_tree"]["sha256"],
    )
    monkeypatch.setattr(stage_module, "R195_DECK_CARDS_SHA256", "sha256:" + "d" * 64)
    monkeypatch.setattr(
        stage_module,
        "_deck_cards_sha256",
        lambda path: stage_module.R195_DECK_CARDS_SHA256,
    )
    monkeypatch.setattr(stage_module, "_validate_r195_matchup_tree", lambda path: None)
    candidate_contract.write_text(
        json.dumps(
            {
                "parent": {"sha256": stage_module.PARENT_SHA256},
                "planner": {"max_neural_passes": 256},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = package_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.recursive_turn_planner."
                    "r198_evaluation_candidate_snapshot/v1"
                ),
                "status": "sealed",
                "no_symlinks": True,
                "all_paths_read_only": True,
                "candidate_id": stage_module.R198_CANDIDATE_ID,
                "candidate_contract_sha256": stage_module.R198_CANDIDATE_CONTRACT_SHA256,
                "package_root": str(package_root),
                "artifacts": artifacts,
                "deck_cards_sha256": stage_module.R195_DECK_CARDS_SHA256,
                "matchup_tree_sha256": stage_module.R195_MATCHUP_TREE_SHA256,
            }
        ),
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    package_root.chmod(0o555)
    evaluation_snapshot = {
        "root": str(snapshot_root),
        "generated_artifacts": {
            "candidate_snapshot": stage_module._file_identity(
                manifest_path, label="candidate manifest"
            )
        },
    }
    candidate = {
        "sidecar": artifacts["sidecar"],
        "sidecar_receipt": artifacts["sidecar_receipt"],
        "completion_receipt": live_identity,
        "candidate_contract": stage_module._file_identity(
            candidate_contract, label="candidate contract"
        ),
    }
    try:
        binding = stage_module._candidate_snapshot_binding(
            evaluation_snapshot, candidate
        )
        assert binding["completion_receipt"] == artifacts["completion_receipt"]
        assert binding["completion_receipt"]["path"] != live_identity["path"]
        assert stat.S_IMODE(live_receipt.lstat().st_mode) == 0o664
        assert stat.S_IMODE(artifact_paths["completion_receipt"].lstat().st_mode) == 0o444

        changed = dict(candidate)
        changed["completion_receipt"] = {**live_identity, "bytes": live_identity["bytes"] + 1}
        with pytest.raises(stage_module.StageError, match="completion receipt differs"):
            stage_module._candidate_snapshot_binding(evaluation_snapshot, changed)
    finally:
        package_root.chmod(0o755)


@pytest.mark.unit
def test_materializer_receives_snapshot_completion_not_live_writable_receipt(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "alakazam-r198-three-arm-eval-attempt10"
    output_root.mkdir()
    live_receipt = tmp_path / "live-candidate" / "r197-receipt.json"
    live_receipt.parent.mkdir()
    live_receipt.write_bytes(b"same immutable receipt bytes")
    live_receipt.chmod(0o664)
    frozen_receipt = (
        tmp_path
        / "source-snapshot"
        / "evaluation-artifacts"
        / "r197-candidate"
        / "r197-completion-receipt.json"
    )
    _write_readonly_identity(
        stage_module, frozen_receipt, live_receipt.read_bytes()
    )
    captured: dict[str, object] = {}

    class _Materializer:
        @staticmethod
        def materialize_r198_evaluation_inputs(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return {"status": "sealed-test-inputs"}

    monkeypatch.setattr(stage_module, "_load_input_materializer", lambda: _Materializer)
    monkeypatch.setattr(
        stage_module,
        "_materialized_inputs_binding",
        lambda raw, **kwargs: dict(raw),
    )
    inputs = {
        "candidate": {"completion_receipt": {"path": str(live_receipt)}},
        "candidate_snapshot": {
            "completion_receipt": {"path": str(frozen_receipt)}
        },
        "official_control_panel": {"registry": {"path": "/sealed/registry.json"}},
        "true_rng_pairing": {"receipt": {"path": "/sealed/capability.json"}},
    }
    base_spec = {"base_spec": {"path": "/sealed/base-spec.json"}}

    stage_module._materialize_evaluation_inputs(
        output_root=output_root, inputs=inputs, base_spec=base_spec
    )

    assert captured["completion_receipt"] == str(frozen_receipt)
    assert captured["completion_receipt"] != str(live_receipt)
    assert stat.S_IMODE(live_receipt.lstat().st_mode) == 0o664
    assert stat.S_IMODE(frozen_receipt.lstat().st_mode) == 0o444


@pytest.mark.unit
def test_stage_rejects_symlinked_output_root_or_child(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "legacy-r195"
    legacy.mkdir()
    root = tmp_path / "alakazam-r198-three-arm-eval-attempt10"
    root.symlink_to(legacy, target_is_directory=True)
    monkeypatch.setattr(stage_module, "DEFAULT_OUTPUT_ROOT", root)

    with pytest.raises(RuntimeError, match="symlink"):
        stage_module._output_root(root, create=False)

    root.unlink()
    root.mkdir()
    child = root / "evaluations"
    child.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        stage_module._output_child(root, "evaluations", "r198-test", label="test child")


@pytest.mark.unit
def test_stage_requires_the_full_uuid_before_importing_torch(
    stage_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    with pytest.raises(RuntimeError, match="full Blackwell UUID"):
        stage_module._verify_blackwell_device("cuda:0")


@pytest.mark.unit
def test_r175_guard_queries_the_user_manager_exactly(
    stage_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        observed.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "LoadState=loaded\nActiveState=failed\nSubState=failed\nMainPID=0\n"
                "Result=exit-code\nExecMainCode=1\nExecMainStatus=143\nNRestarts=2\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(stage_module.subprocess, "run", fake_run)
    state = stage_module._service_properties(
        "pokebot-final-format-alakazam-rtp-r175-rl.service"
    )

    assert observed and observed[0][:3] == ["systemctl", "--user", "show"]
    assert state == stage_module.R175_TERMINAL_SERVICE_STATES[
        "pokebot-final-format-alakazam-rtp-r175-rl.service"
    ]


@pytest.mark.unit
def _r197_snapshot_fixture(
    stage_module,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    schema: str | None = None,
    validator_status: str = "valid",
) -> tuple[Path, Path]:
    root.chmod(stage_module.R197_SOURCE_ROOT_MODE)
    tree_sha = "sha256:" + "1" * 64
    unit_sha = "sha256:" + "2" * 64
    helper = root / "scripts" / "stage_alakazam_rtp_r197_source_snapshot.py"
    helper.parent.mkdir(parents=True)
    returned = {
        "status": validator_status,
        "source_tree_sha256": tree_sha,
        "manifest_sha256": "placeholder",
        "rendered_unit_sha256": unit_sha,
    }
    helper.write_text(
        "def validate_published_root(root):\n"
        f"    return {returned!r}\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    helper_entry = {
        "path": "scripts/stage_alakazam_rtp_r197_source_snapshot.py",
        "type": "file",
        "mode": 0o755,
        "size": helper.stat().st_size,
        "sha256": stage_module._sha256(helper),
    }
    manifest = root / stage_module.R197_SOURCE_MANIFEST_NAME
    manifest_payload = {
        "schema": schema or stage_module.R197_SOURCE_SCHEMA,
        "source_tree_sha256": tree_sha,
        "source_entries": [helper_entry],
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest.chmod(stage_module.R197_SOURCE_MANIFEST_MODE)
    manifest_sha = stage_module._sha256(manifest)
    # Rebind the fixture validator's returned manifest identity, then rebuild
    # its now-changed source entry and the pinned manifest once.
    returned["manifest_sha256"] = manifest_sha
    helper.chmod(0o755)
    helper.write_text(
        "def validate_published_root(root):\n"
        f"    return {returned!r}\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    manifest_payload["source_entries"][0].update(
        {
            "size": helper.stat().st_size,
            "sha256": stage_module._sha256(helper),
        }
    )
    manifest.chmod(0o644)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest.chmod(stage_module.R197_SOURCE_MANIFEST_MODE)
    manifest_sha = stage_module._sha256(manifest)
    # The validator reports the final pinned manifest SHA.  Rewriting it would
    # recurse into its own source identity, so tests patch only this returned
    # scalar after the exact helper-authentication path has executed.
    monkeypatch.setattr(stage_module, "R197_SOURCE_TREE_SHA256", tree_sha)
    monkeypatch.setattr(stage_module, "R197_SOURCE_UNIT_SHA256", unit_sha)
    monkeypatch.setattr(stage_module, "R197_SOURCE_MANIFEST_SHA256", manifest_sha)
    real_loader = stage_module._load_snapshot_validator

    def load_with_final_manifest(root_path, manifest_object):  # type: ignore[no-untyped-def]
        validator, identity = real_loader(root_path, manifest_object)
        # The real stage executes this immutable helper once per process.  This
        # fixture exercises several independent candidate bindings in one test
        # interpreter, so discard only its successful private test module once
        # the callable has retained its globals.
        module_name = _r197_validator_module_name(stage_module, helper)
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            del sys.modules[module_name]

        def wrapped(validated_root):  # type: ignore[no-untyped-def]
            value = dict(validator(validated_root))
            value["manifest_sha256"] = manifest_sha
            return value

        return wrapped, identity

    monkeypatch.setattr(stage_module, "_load_snapshot_validator", load_with_final_manifest)
    return manifest, helper


@pytest.mark.unit
def test_candidate_source_binding_authenticates_manifest_before_validator(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manifest owns schema and pins the exact validator bytes."""

    snapshot_root = tmp_path.resolve()
    _r197_snapshot_fixture(stage_module, monkeypatch, snapshot_root)

    binding = stage_module._candidate_source_binding(snapshot_root)

    assert binding["verification_status"] == "valid"
    assert binding["source_tree_sha256"] == stage_module.R197_SOURCE_TREE_SHA256


@pytest.mark.unit
def test_candidate_source_binding_rejects_a_wrong_manifest_schema(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path.resolve()
    _r197_snapshot_fixture(stage_module, monkeypatch, snapshot_root, schema="wrong")

    with pytest.raises(stage_module.StageError, match="manifest identity changed"):
        stage_module._candidate_source_binding(snapshot_root)


@pytest.mark.unit
def test_candidate_source_binding_rejects_tampered_validator_source(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path.resolve()
    _, helper = _r197_snapshot_fixture(stage_module, monkeypatch, snapshot_root)
    helper.write_text("raise RuntimeError('tampered')\n", encoding="utf-8")
    helper.chmod(0o755)

    with pytest.raises(stage_module.StageError, match="differs from its pinned manifest"):
        stage_module._candidate_source_binding(snapshot_root)


@pytest.mark.unit
def test_candidate_source_binding_requires_validator_status_valid(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path.resolve()
    _r197_snapshot_fixture(
        stage_module,
        monkeypatch,
        snapshot_root,
        validator_status="invalid",
    )

    with pytest.raises(stage_module.StageError, match="source snapshot identity changed"):
        stage_module._candidate_source_binding(snapshot_root)


@pytest.mark.unit
def test_candidate_source_binding_accepts_only_preserved_historical_modes(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path.resolve()
    manifest, helper = _r197_snapshot_fixture(stage_module, monkeypatch, snapshot_root)

    assert stat.S_IMODE(snapshot_root.lstat().st_mode) == stage_module.R197_SOURCE_ROOT_MODE
    assert stat.S_IMODE(manifest.lstat().st_mode) == stage_module.R197_SOURCE_MANIFEST_MODE
    assert stat.S_IMODE(helper.lstat().st_mode) == stage_module.R197_SOURCE_VALIDATOR_MODE
    assert stage_module._candidate_source_binding(snapshot_root)["verification_status"] == "valid"


@pytest.mark.unit
def test_candidate_source_binding_rejects_mode_drift_even_with_pinned_bytes(
    stage_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path.resolve()
    manifest, helper = _r197_snapshot_fixture(stage_module, monkeypatch, snapshot_root)

    snapshot_root.chmod(0o750)
    with pytest.raises(stage_module.StageError, match="root mode changed"):
        stage_module._candidate_source_binding(snapshot_root)

    snapshot_root.chmod(stage_module.R197_SOURCE_ROOT_MODE)
    manifest.chmod(0o444)
    with pytest.raises(stage_module.StageError, match="mode changed"):
        stage_module._candidate_source_binding(snapshot_root)

    manifest.chmod(stage_module.R197_SOURCE_MANIFEST_MODE)
    helper.chmod(0o744)
    with pytest.raises(stage_module.StageError, match="mode changed"):
        stage_module._candidate_source_binding(snapshot_root)


@pytest.mark.unit
def test_one_shot_unit_is_uuid_pinned_and_cannot_control_r175() -> None:
    unit = UNIT.read_text(encoding="utf-8")

    assert "Type=oneshot" in unit
    assert "Restart=no" in unit
    assert "CUDA_VISIBLE_DEVICES=GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6" in unit
    assert "CG_LIB_PATH=/home/pokebot/poke-bot-agent-deployments/final-format-alakazam-rtp-r198-three-arm-eval-v1/kaggle/input/rtp-eval-cg" in unit
    assert "eval-cg-closure.json" in unit
    assert "r198-eval-cg-closure-manifest.json" not in unit
    assert "rtp-pairing-v2-probes-canonical-seal-v2/true-rng-pairing-capability-v2.json" in unit
    assert "state/matchup_adapter_roster.json" in unit
    assert "systemctl" not in unit
    assert not any(
        line.startswith(prefix)
        for line in unit.splitlines()
        for prefix in ("Conflicts=", "BindsTo=", "PartOf=", "Requires=")
    )
