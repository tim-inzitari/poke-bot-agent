import json
from pathlib import Path

import poke_bot.alakazam_rule_derivative_readiness_rev9 as readiness


def test_readiness_assessment_is_fail_closed_and_nonactivating(tmp_path: Path, monkeypatch):
    paths = {}
    expected = {}
    for name in ("r195", "r274", "corpus", "branch"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}")
        paths[name] = path
        expected[name] = readiness.sha256_file(path)
    branch_payload = {
        "eligible_trainable_branches": ["public_rule_semantic_projection"],
        "candidate_training_allowed": False,
    }
    paths["branch"].write_text(json.dumps(branch_payload))
    expected["branch"] = readiness.sha256_file(paths["branch"])
    monkeypatch.setattr(readiness, "R195_SHA256", expected["r195"])
    monkeypatch.setattr(readiness, "R274_ITER1_SHA256", expected["r274"])
    monkeypatch.setattr(readiness, "CORPUS_MANIFEST_SHA256", expected["corpus"])
    monkeypatch.setattr(readiness, "BRANCH_RECEIPT_SHA256", expected["branch"])
    monkeypatch.setattr(readiness, "_checkpoint_inventory", lambda _path: {
        "rl_iteration": 1,
        "optimizer_state_embedded": True,
    })
    monkeypatch.setattr(readiness, "_gpu_inventory", lambda: [{
        "torch_ordinal": 0,
        "name": "NVIDIA RTX PRO 5000 Blackwell",
        "total_memory_bytes": 1,
        "compute_capability": [12, 0],
    }])
    monkeypatch.setattr(readiness, "_systemctl_show", lambda unit: {
        "unit": unit,
        "active_state": "active",
        "sub_state": "running",
        "main_pid": 42,
        "exec_main_status": 0,
        "fragment_path": "/unit",
        "unit_file_state": "enabled",
    })
    result = readiness.build_readiness_assessment(
        r195_checkpoint_path=paths["r195"],
        r274_checkpoint_path=paths["r274"],
        corpus_manifest_path=paths["corpus"],
        branch_receipt_path=paths["branch"],
        exact_pause_services=(
            "pokebot-alakazam-r274-rl.service",
            "pokebot-alakazam-r274-rl-submission-boundaries.service",
        ),
    )
    assert result["readiness_passed"] is False
    assert result["service_control_performed"] is False
    assert result["training_or_activation_performed"] is False
    assert result["eligible_trainable_branches"] == ["public_rule_semantic_projection"]
    assert any("predecessor_service_not_inactive" in blocker for blocker in result["blockers"])
    assert "contract_cuda_1_does_not_match_torch_blackwell_ordinal:0" in result["blockers"]


def test_create_only_assessment(tmp_path: Path):
    path = tmp_path / "assessment.json"
    digest = readiness.write_create_only(path, {"readiness_passed": False})
    assert digest == readiness.sha256_file(path)
    assert path.stat().st_mode & 0o222 == 0
