"""Focused CPU-only coverage for the r244 exact staged fault harness."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_r244_exact_staged_fault_harness.py"


def _load_module() -> Any:
    name = "r244_exact_staged_fault_harness_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_one_json_line(sock: socket.socket) -> dict[str, object]:
    buffer = bytearray()
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        assert chunk
        buffer.extend(chunk)
    raw, _sep, _rest = buffer.partition(b"\n")
    value = json.loads(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def test_fake_child_is_a_real_socket_process_and_never_claims_search() -> None:
    module = _load_module()
    parent, child = socket.socketpair()
    try:
        child_fd = child.detach()
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(SCRIPT),
                "--fake-child",
                "--child-fd",
                str(child_fd),
                "--fault-class",
                "evaluator",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(child_fd,),
        )
        os.close(child_fd)
        ready = _read_one_json_line(parent)
        assert ready["type"] == "ready"
        assert ready["fault_harness"] is True
        parent.sendall(
            (
                json.dumps(
                    {
                        "schema": module._FAKE_CHILD_READY_SCHEMA,
                        "type": "select",
                        "request_id": 7,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        error = _read_one_json_line(parent)
        assert error["type"] == "error"
        assert error["code"] == "fault_injected_evaluator"
        assert error["detail"] == {"fault_harness": True, "fault_class": "evaluator"}
    finally:
        parent.close()
        if "process" in locals() and process.poll() is None:
            process.terminate()
            process.wait(timeout=1.0)


def test_install_hook_targets_only_loaded_staged_broker_and_reaps_its_exact_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    stage = tmp_path / "stage"
    broker_file = stage / "poke_bot/r228_kaggle_broker.py"
    broker_file.parent.mkdir(parents=True)
    broker_file.write_text("# synthetic module location for harness hook\n", encoding="utf-8")
    fake_subprocess = types.SimpleNamespace(Popen=subprocess.Popen)
    fake_broker = types.SimpleNamespace(__file__=str(broker_file), subprocess=fake_subprocess)
    monkeypatch.setitem(sys.modules, "poke_bot.r228_kaggle_broker", fake_broker)
    handle = module.install_fault_injected_broker_child(stage, "cleanup")
    parent, child = socket.socketpair()
    try:
        child_fd = child.detach()
        process = fake_subprocess.Popen(
            ["ignored-by-injection"],
            env=dict(os.environ),
            pass_fds=(child_fd,),
        )
        os.close(child_fd)
        assert process.pid == handle.spawned[0].pid
        ready = _read_one_json_line(parent)
        assert ready["type"] == "ready"
        assert ready["preload_stock_library"] == module._FAKE_CHILD_PRELOAD
    finally:
        parent.close()
        handle.restore()
        cleanup = handle.cleanup_owned_children()
    assert fake_subprocess.Popen is subprocess.Popen
    assert cleanup and all(row["reaped"] for row in cleanup)
    assert cleanup[0]["pid"] == process.pid


def test_unreaped_proxy_forwards_exact_term_kill_then_reports_hard_branch() -> None:
    module = _load_module()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proxy = module._ControlledUnreapedProxy(process)
    try:
        proxy.terminate()
        proxy.kill()
        with pytest.raises(subprocess.TimeoutExpired):
            proxy.wait(timeout=0.001)
    finally:
        cleanup = proxy.reap_underlying()
    assert proxy.poll() is None
    assert proxy.terminate_calls == 1
    assert proxy.kill_calls == 1
    assert cleanup["reaped"] is True
    assert cleanup["pid"] == process.pid


def test_receipt_uses_binder_common_identity_and_all_required_fault_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    stage = tmp_path / "stage"
    stage.mkdir()
    snapshot = {"members": {"main.py": "sha256:one"}, "tree_sha256": "sha256:tree"}
    common = {
        "candidate_archive_sha256": "sha256:archive",
        "candidate_archive_size_bytes": 123,
        "member_manifest_sha256": "sha256:manifest",
        "entrypoint_sha256": "sha256:entrypoint",
        "r225_contract_sha256": module.R246_TYPED_CONTRACT_SHA256,
        "canonical_libcg_contract_sha256": "sha256:r236",
        "linux_x86_64_libcg_sha256": "sha256:linux",
        "linux_x86_64_libcg_size_bytes": 7,
        "complete_ordered_action_cap": 65_536,
        "simulator_search_lane_count": 2,
        "phase1_submission_environment": {"hdd_space_gib": 11.8},
        "r240_hybrid_scheduler": {"high_confidence_threshold": 0.8},
        "deterministic_continuation": {"max_depth": 8},
    }
    monkeypatch.setattr(
        module,
        "load_binding_identity",
        lambda **_kwargs: {
            "common_identity": common,
            "exact_package": {"stage": str(stage)},
            "stage_contract": {"manifest": {}},
        },
    )
    monkeypatch.setattr(module, "stage_snapshot", lambda _stage: dict(snapshot))
    monkeypatch.setattr(
        module,
        "_run_owned_worker",
        lambda *, case, **_kwargs: (
            {
                "status": "passed",
                "full_gameplay_success_marker_count": 0,
                "degraded_marker_count": 1,
            }
            if case in module.FAULT_CLASSES
            else {"status": "passed", "case": case}
        ),
    )
    output = tmp_path / "focused.json"
    receipt = module.run_fault_harness(
        stage=stage,
        candidate_archive=tmp_path / "candidate.tar.gz",
        member_manifest=tmp_path / "manifest.json",
        output=output,
    )
    assert receipt["schema"] == module.PREFLIGHT_RECEIPT_SCHEMA
    assert receipt["receipt_name"] == module.RECEIPT_NAME
    assert receipt["fault_classes_covered"] == list(module.FAULT_CLASSES)
    assert receipt["focused_fault_suite_passed"] is True
    assert receipt["nonreaped_child_hard_fail_test_passed"] is True
    assert receipt["parent_returned_action_legality_hard_fail_test_passed"] is True
    assert receipt["fault_injected_full_game_degraded_marker_and_no_viability_credit_passed"] is True
    assert receipt["harness"]["r246_canonical_contract_binding"] == {
        "owner_decision_revision": 246,
        "r225_contract_sha256": module.R246_TYPED_CONTRACT_SHA256,
        "terminal_win_proof_exercised_by_fault_suite": False,
        "faulted_game_viability_success_allowed": False,
    }
    assert output.exists() and output.stat().st_mode & 0o222 == 0
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["candidate_archive_sha256"] == "sha256:archive"
    assert "terminal_win_proof" not in persisted


def test_refuses_to_publish_when_one_faulted_game_claims_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        module,
        "load_binding_identity",
        lambda **_kwargs: {
            "common_identity": {"r225_contract_sha256": module.R246_TYPED_CONTRACT_SHA256},
            "exact_package": {},
            "stage_contract": {},
        },
    )
    monkeypatch.setattr(module, "stage_snapshot", lambda _stage: {"tree": "same"})

    def worker(*, case: str, **_kwargs: Any) -> dict[str, Any]:
        if case == "native":
            return {
                "status": "passed",
                "degraded_marker_count": 1,
                "full_gameplay_success_marker_count": 1,
            }
        if case in module.FAULT_CLASSES:
            return {
                "status": "passed",
                "degraded_marker_count": 1,
                "full_gameplay_success_marker_count": 0,
            }
        return {"status": "passed"}

    monkeypatch.setattr(module, "_run_owned_worker", worker)
    with pytest.raises(module.FaultHarnessError, match="viability success"):
        module.run_fault_harness(
            stage=stage,
            candidate_archive=tmp_path / "candidate.tar.gz",
            member_manifest=tmp_path / "manifest.json",
            output=tmp_path / "must-not-exist.json",
        )


def test_refuses_a_common_identity_that_is_not_final_r246(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        module,
        "load_binding_identity",
        lambda **_kwargs: {
            "common_identity": {"r225_contract_sha256": "sha256:old"},
            "exact_package": {},
            "stage_contract": {},
        },
    )
    with pytest.raises(module.FaultHarnessError, match="common identity r225 digest"):
        module.run_fault_harness(
            stage=stage,
            candidate_archive=tmp_path / "candidate.tar.gz",
            member_manifest=tmp_path / "manifest.json",
            output=tmp_path / "must-not-exist.json",
        )


def test_worker_result_parser_rejects_missing_or_multiple_result_rows() -> None:
    module = _load_module()
    with pytest.raises(module.FaultHarnessError, match="exactly one"):
        module._parse_worker_result("nothing\n")
    row = module.WORKER_RESULT_PREFIX + json.dumps({"status": "passed"})
    with pytest.raises(module.FaultHarnessError, match="exactly one"):
        module._parse_worker_result(row + "\n" + row)
