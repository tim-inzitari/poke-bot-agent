"""Focused fail-closed checks for the r197 submission RTP profiles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot import rtp_evaluation_promotion as promotion


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "submission" / "main.py"


def _load_submission_module(stage: Path):
    spec = importlib.util.spec_from_file_location(
        f"submission_rtp_binding_{hash(stage)}", MAIN
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stage(tmp_path: Path) -> tuple[Path, object, str]:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "deck.csv").write_text("".join("1\n" for _ in range(60)))
    model_bytes = b"r197-model-bytes"
    (stage / "model.pt").write_bytes(model_bytes)
    module = _load_submission_module(stage)
    return stage, module, _sha256(model_bytes)


def _write_profile(stage: Path, value: dict[str, object]) -> None:
    (stage / "runtime_profile.json").write_text(json.dumps(value))


def _seal_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)
    return path


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def _masked_packaged_evaluation(stage: Path) -> Path:
    """Write the sealed r198 review receipt for the current hard hold."""

    registry = stage / "research_control_registry_v1.json"
    registry.write_bytes((ROOT / "ops" / "research_control_registry_v1.json").read_bytes())
    os.chmod(registry, 0o444)
    results = _seal_json(stage / "evaluation-results.json", {"rows": []})
    payload: dict[str, object] = {
        "schema": promotion.EVALUATION_RECEIPT_SCHEMA,
        "status": "ready_for_separate_promotion_review",
        "created_at_utc": "2026-08-09T00:00:00Z",
        "promotion_decision": {
            "eligible_for_separate_promotion_review": True,
            "self_promotion_performed": False,
            "serving_change_authorized": False,
        },
        "evaluation_isolation": {
            "training_eligible": False,
            "replay_eligible": False,
            "formal_gate": False,
            "serving_change_authorized": False,
            "self_promotion_allowed": False,
        },
        "results": {**_identity(results), "in_memory": False},
        "promotion_gates": {
            name: {"passed": True} for name in promotion._REQUIRED_GATES
        },
        "frozen_artifacts": {
            "opponents": [
                {"id": opponent, "content_digest": digest}
                for opponent, digest in promotion.R198_OFFICIAL_CONTROL_OPPONENTS.items()
            ]
        },
        "official_control_panel": {
            "registry": _identity(registry),
            "opponents": dict(promotion.R198_OFFICIAL_CONTROL_OPPONENTS),
        },
        "r197_source_exclusion_binding": {
            "candidate_contract_sha256": promotion.R198_CANDIDATE_CONTRACT_SHA256,
            "r197_source_disjoint": True,
            "evaluation_only": True,
            "source_identity_overlap_count": 0,
            "candidate_target_status": "masked_absent_no_fabrication",
            "trusted_counterfactual_candidate_targets_available": False,
        },
    }
    payload["receipt_input_sha256"] = promotion.canonical_digest(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at_utc", "receipt_input_sha256"}
        }
    )
    return _seal_json(stage / "rtp_evaluation_receipt.json", payload)


@pytest.mark.unit
def test_explicit_off_binds_model_and_scrubs_inherited_rtp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, module, model_digest = _stage(tmp_path)
    _write_profile(
        stage,
        {
            "schema": "poke_bot.submission_runtime_profile/v1",
            "rtp_mode": "off",
            "recursive_turn_planner": "disabled",
            "display": "NO RTP",
            "rtp_sidecar_packaged": False,
            "model_checkpoint_sha256": model_digest,
        },
    )
    monkeypatch.chdir(stage)
    monkeypatch.setenv("POKEBOT_USE_RECURSIVE_TURN_PLANNER", "1")
    monkeypatch.setenv("POKEBOT_RTP_CHECKPOINT", "/host/stale.pt")
    profile = module._apply_runtime_profile()

    assert profile["rtp_mode"] == "off"
    assert module._runtime_profile_mode(profile) == "off"
    assert module.os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "0"
    assert "POKEBOT_RTP_CHECKPOINT" not in module.os.environ


@pytest.mark.unit
def test_direct_profile_is_bridge_only_and_binds_inert_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, module, model_digest = _stage(tmp_path)
    sidecar = stage / "rtp_shadow_planner.pt"
    sidecar.write_bytes(b"inert-r197-sidecar")
    _write_profile(
        stage,
        {
            "schema": "poke_bot.submission_runtime_profile/v1",
            "rtp_mode": "direct",
            "recursive_turn_planner": "enabled",
            "display": "DIRECT RTP",
            "rtp_sidecar_packaged": True,
            "rtp_direct_bridge_only": True,
            "rtp_sizing_profile": "pure_rl_r197",
            "specialist_id": "alakazam",
            "model_checkpoint_sha256": model_digest,
            "parent_checkpoint_sha256": model_digest,
            "rtp_checkpoint_sha256": module._sha256_file(sidecar),
            "rtp_config_sha256": _sha256(b"r197-config"),
            "max_neural_passes": 256,
            "max_action_combos": 1024,
            "required_neural_passes": {"normal": 6, "forced_replan": 5},
        },
    )
    monkeypatch.chdir(stage)
    monkeypatch.setenv("POKEBOT_RTP_CHECKPOINT", "/host/stale.pt")
    module._apply_runtime_profile()

    assert module.os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "1"
    assert "POKEBOT_RTP_ALLOW_UNTRAINED" not in module.os.environ
    assert module.os.environ["POKEBOT_RTP_SIZING_PROFILE"] == "pure_rl_r197"
    assert module.os.environ["POKEBOT_RTP_CHECKPOINT"] == str(sidecar.resolve())
    assert module.os.environ["POKEBOT_RTP_PARENT_CHECKPOINT_SHA256"] == model_digest
    assert "POKEBOT_RTP_SERVING_QUALIFIED" not in module.os.environ
    assert "POKEBOT_RTP_PROMOTION_RECEIPT" not in module.os.environ

    plan_turn_calls: list[object] = []

    def original_plan_turn(
        _memory: object,
        *,
        policy_logits: object = None,
        force_recurse: object = None,
    ) -> str:
        del policy_logits
        plan_turn_calls.append(force_recurse)
        return "direct-policy"

    planner = SimpleNamespace(plan_turn=original_plan_turn)
    bridge = SimpleNamespace(planner=planner)
    module._force_direct_rtp_bridge(SimpleNamespace(_rtp_bridge=bridge))
    recurse, diagnostics = planner.should_recurse(object(), policy_logits=object())
    assert recurse is False
    assert diagnostics["recursion_forced_disabled"] is True
    assert bridge._submission_direct_only is True
    assert planner.plan_turn(object(), force_recurse=True) == "direct-policy"
    assert plan_turn_calls == [None]

    invalid_profile = json.loads((stage / "runtime_profile.json").read_text())
    invalid_profile["max_action_combos"] = 256
    _write_profile(stage, invalid_profile)
    with pytest.raises(RuntimeError, match="exact 1024 action combinations"):
        module._apply_runtime_profile()

    invalid_profile["max_action_combos"] = 1024
    invalid_profile["required_neural_passes"] = {"normal": 5, "forced_replan": 5}
    _write_profile(stage, invalid_profile)
    with pytest.raises(RuntimeError, match="normal=6/forced-replan=5"):
        module._apply_runtime_profile()


@pytest.mark.unit
def test_new_profile_refuses_legacy_mode_alias(tmp_path: Path) -> None:
    _stage_path, module, _model_digest = _stage(tmp_path)
    with pytest.raises(ValueError, match="unsupported submitted"):
        module._runtime_profile_mode(
            {
                "schema": "poke_bot.submission_runtime_profile/v1",
                "rtp_mode": "enabled",
                "recursive_turn_planner": "enabled",
            }
        )


@pytest.mark.unit
def test_r197_sidecar_config_requires_exact_256(
    tmp_path: Path,
) -> None:
    _stage_path, module, _model_digest = _stage(tmp_path)
    base = {
        "schema": "poke_bot.recursive_turn_planner/v1",
        "sizing_profile": "pure_rl_r197",
        "d_model": 96,
        "dynamics_width": 192,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_neural_passes": 256,
        "max_plan_length": 12,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "prefer_option_hidden": True,
    }
    module._assert_r197_sidecar_config(base, expected_max_neural_passes=256)

    stale = dict(base, max_neural_passes=24)
    with pytest.raises(RuntimeError, match="max_neural_passes"):
        module._assert_r197_sidecar_config(stale, expected_max_neural_passes=256)
    superseded = dict(base, max_neural_passes=32)
    with pytest.raises(RuntimeError, match="revision-198 exact neural-pass budget"):
        module._assert_r197_sidecar_config(superseded, expected_max_neural_passes=32)
    excessive = dict(base, max_neural_passes=257)
    with pytest.raises(RuntimeError, match="hard ceiling"):
        module._assert_r197_sidecar_config(excessive, expected_max_neural_passes=257)


@pytest.mark.unit
def test_r197_startup_requires_exact_1024_action_combinations(
    tmp_path: Path,
) -> None:
    _stage_path, module, _model_digest = _stage(tmp_path)
    config = SimpleNamespace()
    planner = SimpleNamespace(config=config)
    bridge = SimpleNamespace(
        planner=planner,
        config=config,
        max_action_combos=1024,
    )
    module._assert_live_recursive_config(
        SimpleNamespace(_rtp_bridge=bridge), {}
    )

    bridge.max_action_combos = 256
    with pytest.raises(RuntimeError, match="exact 1024 action combinations"):
        module._assert_live_recursive_config(
            SimpleNamespace(_rtp_bridge=bridge), {}
        )


@pytest.mark.unit
def test_explicit_profile_fails_when_model_bytes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, module, model_digest = _stage(tmp_path)
    _write_profile(
        stage,
        {
            "schema": "poke_bot.submission_runtime_profile/v1",
            "rtp_mode": "off",
            "recursive_turn_planner": "disabled",
            "display": "NO RTP",
            "rtp_sidecar_packaged": False,
            "model_checkpoint_sha256": model_digest,
        },
    )
    (stage / "model.pt").write_bytes(b"tampered")
    monkeypatch.chdir(stage)
    with pytest.raises(RuntimeError, match="model digest"):
        module._apply_runtime_profile()


@pytest.mark.unit
def test_recursive_profile_requires_external_promotion_for_inert_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT))
    from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
    from poke_bot.recursive_turn_planner.profiles import get_profile
    from poke_bot.recursive_turn_planner.training.checkpoint import _serialized_config

    stage, module, model_digest = _stage(tmp_path)
    tree = stage / "matchup_tree.json"
    tree.write_text('{"r197":true}\n')
    evaluation = _masked_packaged_evaluation(stage)

    config = _serialized_config(get_profile("pure_rl_r197").to_config())
    planner = RecursiveTurnPlanner(get_profile("pure_rl_r197").to_config())
    sidecar = stage / "rtp_shadow_planner.pt"
    torch.save(
        {
            "schema": "poke_bot.recursive_turn_planner.shadow_train/v1",
            "research_only": False,
            "shadow_only": True,
            # The external receipt, not the sidecar, grants serving authority.
            "serving_eligible": False,
            "action_authority_enabled": False,
            "parent_checkpoint_sha256": model_digest,
            "config": config,
            "state_dict": planner.state_dict(),
        },
        sidecar,
    )
    promotion = {
        "schema": "poke_bot.rtp_promotion/v1",
        "status": "accepted",
        "specialist_id": "alakazam",
        "parent_checkpoint_sha256": model_digest,
        "sidecar_sha256": module._sha256_file(sidecar),
        "sidecar_config_sha256": module._canonical_json_sha256(config),
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "required_neural_passes": {"normal": 6, "forced_replan": 5},
        "deck_file_sha256": module._sha256_file(stage / "deck.csv"),
        "deck_cards_sha256": module._deck_cards_sha256(stage / "deck.csv"),
        "matchup_tree_sha256": module._sha256_file(tree),
        "evaluation_receipt_path": str(evaluation),
        "evaluation_receipt_sha256": module._sha256_file(evaluation),
        "identity_gate_passed": True,
        "planner_activation_gate_passed": True,
        "reliability_gate_passed": True,
        "heldout_efficacy_gate_passed": True,
        "robustness_gate_passed": True,
        "latency_gate_passed": True,
        "serving_eligible": True,
        "action_authority_enabled": True,
        "created_at_utc": "2026-08-09T00:00:00Z",
    }
    packaged_promotion = stage / "rtp_promotion_receipt.json"
    _seal_json(packaged_promotion, promotion)
    _write_profile(
        stage,
        {
            "schema": "poke_bot.submission_runtime_profile/v1",
            "rtp_mode": "recursive",
            "recursive_turn_planner": "enabled",
            "display": "RTP",
            "rtp_sidecar_packaged": True,
            "rtp_sizing_profile": "pure_rl_r197",
            "specialist_id": "alakazam",
            "model_checkpoint_sha256": model_digest,
            "parent_checkpoint_sha256": model_digest,
            "rtp_checkpoint_sha256": module._sha256_file(sidecar),
            "rtp_config_sha256": module._canonical_json_sha256(config),
            "max_neural_passes": 256,
            "max_action_combos": 1024,
            "required_neural_passes": {"normal": 6, "forced_replan": 5},
            "deck_file_sha256": module._sha256_file(stage / "deck.csv"),
            "deck_cards_sha256": module._deck_cards_sha256(stage / "deck.csv"),
            "matchup_tree_sha256": module._sha256_file(tree),
            "rtp_promotion_receipt_file": "rtp_promotion_receipt.json",
            "rtp_promotion_receipt_sha256": module._sha256_file(
                packaged_promotion
            ),
            "rtp_evaluation_receipt_file": "rtp_evaluation_receipt.json",
            "rtp_evaluation_receipt_sha256": module._sha256_file(evaluation),
        },
    )
    monkeypatch.chdir(stage)
    profile = module._apply_runtime_profile()
    with pytest.raises(
        RuntimeError,
        match="trusted counterfactual candidate targets are absent",
    ):
        module._assert_recursive_rtp_binding(profile, model_digest=model_digest)

    assert module.os.environ["POKEBOT_RTP_SERVING_QUALIFIED"] == "1"
    assert module.os.environ["POKEBOT_RTP_PROMOTION_RECEIPT"] == str(
        packaged_promotion.resolve()
    )
    assert module.os.environ["POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT"] == str(
        evaluation
    )
