from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.model_registry import sha256
from scripts import run_post_alakazam_starmie_handoff as handoff


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ops/post_alakazam_starmie_handoff_v1.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _temp_contract(tmp_path: Path) -> Path:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["paths"] = {
        key: str((tmp_path / key).resolve()) for key in payload["paths"]
    }
    path = tmp_path / "contract.json"
    _write(path, payload)
    return path


def test_contract_is_exact_pass_gated_and_queues_two_nonblocking_copies() -> None:
    contract, digest = handoff.load_contract(CONTRACT)
    assert digest == sha256(CONTRACT)
    assert contract["deployment_mode"] == "exact_pass_gated_automatic_handoff"
    assert (
        contract["production_mutation_policy"]
        == "forbidden_before_immutable_activation_receipt"
    )
    assert contract["automatic_next_specialist_start_after_activation_receipt"] is True
    assert contract["next_specialist_id"] == "hops-trevenant"
    assert contract["required_kaggle_copies"] == 2
    assert contract["submission_completion_blocks_handoff"] is False
    assert contract["phases"] == list(handoff.PHASES)
    assert contract["families"]["distilled_core"] == handoff.CORE_FAMILY
    assert contract["families"]["starmie_bootstrap"] == handoff.STARMIE_FAMILY


def test_corpus_preflight_fails_before_missing_inputs_can_run(tmp_path: Path) -> None:
    contract, _digest = handoff.load_contract(_temp_contract(tmp_path))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        handoff.validate_corpora(contract)


def test_pending_checkpoint_bound_copies_do_not_block_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _digest = handoff.load_contract(_temp_contract(tmp_path))
    passed_family = handoff.required_path(contract, "passed_family")
    passed_family.mkdir(parents=True)
    _write(passed_family / "manifest.json", {"checkpoint_digest": "unused"})
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"exact bundle")
    bundle_digest = sha256(bundle)
    attestation = tmp_path / "attestation.json"
    _write(
        attestation,
        {"file_sha256": bundle_digest, "go_first_if_offered": True},
    )
    plan = {
        "checkpoint_digest": "sha256:" + "1" * 64,
        "contract_sha256": "sha256:" + "2" * 64,
        "commit_digest": "sha256:" + "3" * 64,
        "commit_file_sha256": "sha256:" + "4" * 64,
        "exact_result_pointer_sha256": "sha256:" + "5" * 64,
        "gate_id": "exact-gate",
    }
    queued = []
    for slot in (1, 2):
        copy = tmp_path / f"copy-{slot}.tar.gz"
        copy.write_bytes(bundle.read_bytes())
        queued.append(
            {
                "copy_number": slot,
                "checkpoint_checksum": plan["checkpoint_digest"],
                "queue_status": "pending",
                "file": str(copy),
                "file_sha256": bundle_digest,
            }
        )
    handler = {
        "phase": "complete_handoff_started",
        "gate": plan,
        "approved_submission_count": 2,
        "automatic_retries": False,
        "submission_mode": "queue_and_continue",
        "all_submissions_succeeded": False,
        "successful_submission_count": 0,
        "submission_bundle": {
            "path": str(bundle),
            "sha256": bundle_digest,
            "attestation": str(attestation),
            "contents": {"model_sha256": plan["checkpoint_digest"]},
        },
        "queued_submissions": queued,
    }
    _write(handoff.required_path(contract, "handler_state"), handler)
    monkeypatch.setattr(
        handoff, "validate_exact_pass", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(
        handoff,
        "verify_frozen_model",
        lambda _path: {
            "checkpoint_digest": plan["checkpoint_digest"],
            "provenance": {
                "contract_sha256": plan["contract_sha256"],
                "commit_digest": plan["commit_digest"],
            },
        },
    )
    evidence = handoff.validate_pass_and_uploads(contract)
    assert evidence["submission_status"] == "pending"
    assert evidence["submission_completion_blocks_handoff"] is False
    assert len(evidence["pending_submissions"]) == 2


def test_pipeline_orders_archive_distillation_bootstrap_then_staged_rl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = _temp_contract(tmp_path)
    events: list[str] = []
    passed = {
        "checkpoint_digest": "sha256:" + "1" * 64,
        "identity_sha256": "sha256:" + "2" * 64,
        "submission_status": "pending",
    }
    corpora = {
        "core": {
            "pointer_sha256": "sha256:" + "3" * 64,
            "manifest_sha256": "sha256:" + "4" * 64,
        },
        "starmie": {
            "pointer_sha256": "sha256:" + "5" * 64,
            "manifest_sha256": "sha256:" + "6" * 64,
        },
    }
    archive = {"receipt_sha256": "sha256:" + "7" * 64}
    core = {"checkpoint_digest": "sha256:" + "8" * 64}
    starmie = {"checkpoint_digest": "sha256:" + "9" * 64}
    monkeypatch.setattr(handoff, "validate_pass_and_uploads", lambda _c: passed)
    monkeypatch.setattr(
        handoff,
        "validate_corpora",
        lambda _c: events.append("corpora-verified") or corpora,
    )
    monkeypatch.setattr(
        handoff,
        "nonblocking_archive_evidence",
        lambda *_a, **_kw: events.append("archive-pending") or archive,
    )
    monkeypatch.setattr(
        handoff,
        "validate_core_ready",
        lambda *_a: events.append("core-verified") or core,
    )
    monkeypatch.setattr(
        handoff,
        "validate_starmie_ready",
        lambda *_a: events.append("starmie-verified") or starmie,
    )
    monkeypatch.setattr(
        handoff,
        "write_or_validate_activation",
        lambda *_a: {"identity_sha256": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(handoff, "verify_rl_start", lambda *_a: {})
    monkeypatch.setattr(handoff, "save_state", lambda *_a, **_kw: None)
    monkeypatch.setattr(handoff, "archive_command", lambda _c: ["archive"])
    monkeypatch.setattr(handoff, "core_command", lambda _c: ["core"])
    monkeypatch.setattr(handoff, "starmie_command", lambda _c: ["starmie"])

    def run(argv: list[str]) -> None:
        events.append("run:" + argv[-1])

    monkeypatch.setattr(handoff, "run_checked", run)
    assert handoff.run_pipeline(contract_path) == 0
    assert events == [
        "archive-pending",
        "corpora-verified",
        "run:core",
        "core-verified",
        "run:starmie",
        "starmie-verified",
        "run:pokebot-pure-rl-trevenant-staged.service",
        "run:pokebot-pure-rl-trevenant-staged.service",
    ]


def test_activation_receipt_digest_is_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = _temp_contract(tmp_path)
    contract, _digest = handoff.load_contract(contract_path)
    identity = {"service": "trevenant", "checkpoint": "sha256:" + "1" * 64}
    monkeypatch.setattr(
        handoff, "deterministic_evidence", lambda *_a, **_kw: (contract, identity)
    )
    receipt_path = handoff.required_path(contract, "activation_receipt")
    _write(
        receipt_path,
        {
            "schema": handoff.ACTIVATION_SCHEMA,
            "status": "ready",
            "identity": identity,
            "identity_sha256": "sha256:" + "0" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="activation evidence"):
        handoff.verify_rl_start(contract_path, check_source=False)


def test_staged_units_use_immutable_root_and_protected_outputs() -> None:
    pipeline = (
        ROOT / "deploy/systemd/pokebot-post-alakazam-gate-pipeline.service"
    ).read_text(encoding="utf-8")
    starmie = (
        ROOT / "deploy/systemd/pokebot-pure-rl-trevenant-staged.service"
    ).read_text(encoding="utf-8")
    immutable_root = "/home/pokebot/poke-bot-agent-deployments/pure-rl-resident-v31-matchup-runtime"
    assert f"WorkingDirectory={immutable_root}" in pipeline
    assert f"WorkingDirectory={immutable_root}" in starmie
    assert "run_post_alakazam_starmie_handoff.py run" in pipeline
    assert "archive_passed_system_to_elmo.py" not in pipeline
    assert "authoritative-all-recognized-latest10" not in pipeline
    protected_trevenant_v2 = (
        "hops-trevenant-v2/PROTECTED_EXPERT_CORPUS.json"
    )
    assert protected_trevenant_v2 in pipeline
    assert protected_trevenant_v2 in starmie
    assert "hops-trevenant/PROTECTED_EXPERT_CORPUS.json" not in pipeline
    assert "hops-trevenant/PROTECTED_EXPERT_CORPUS.json" not in starmie
    assert "StartLimitIntervalSec=0" in pipeline
    assert "Restart=on-failure" in pipeline
    assert "RestartSec=300" in pipeline
    assert "verify-rl-start" in starmie
    assert "hops-trevenant_expert_bootstrap_from_distilled_core_v1/model.pt" in starmie
    assert "--specialist-archetype hops-trevenant" in starmie
    assert "--games-per-iter 8192" in starmie
    assert "--train-epochs 1" in starmie
    assert "--expert-rehearsal-every 5" in starmie
    assert "--expert-rehearsal-epochs 5" in starmie
    assert "--alakazam-guide-loss-weight 0.0" in starmie
    assert "Environment=PURE_RL_SIM_WORKERS=128" in starmie
    assert "Environment=PURE_RL_LEAF_GPU0_REPLICAS=10" in starmie
    assert "Environment=PURE_RL_LEAF_GPU1_REPLICAS=24" in starmie
    assert "MemoryHigh=110G" in starmie
    assert "MemoryMax=116G" in starmie
    assert "[Install]" not in starmie
    # Each predecessor synchronously starts its successor. Ordering against the
    # still-activating caller would deadlock both systemctl transactions.
    assert (
        "After=network-online.target pokebot-passed-gate-handler.service"
        not in pipeline
    )
    assert (
        "After=network-online.target pokebot-post-alakazam-gate-pipeline.service"
        not in starmie
    )


def test_orchestrator_does_not_modify_or_invoke_passed_gate_handler() -> None:
    source = (ROOT / "scripts/run_post_alakazam_starmie_handoff.py").read_text(
        encoding="utf-8"
    )
    assert "handle_passed_gate.py" in source  # documented separation only
    assert "scripts/handle_passed_gate.py" not in source
    assert "submit_one_approved_copy" not in source
    assert "kaggle competitions submit" not in source
