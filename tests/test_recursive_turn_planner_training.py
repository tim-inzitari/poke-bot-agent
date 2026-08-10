"""Unit tests for Recursive Turn Planner / PokeRLM shadow training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from poke_bot.poke_rlm.model_core import PokeRLMModelCore
from poke_bot.poke_rlm.training.losses import compute_poke_rlm_losses
from poke_bot.poke_rlm.training.labels import PlanSupervisionLabels
from poke_bot.poke_rlm.training.shadow_train import (
    PokeRLMTrainConfig,
    load_poke_rlm_core,
    train_poke_rlm_shadow,
)
from poke_bot.poke_rlm.config import PokeRLMConfig
from poke_bot.recursive_turn_planner.training import (
    load_rtp_checkpoint,
    make_synthetic_batches,
    save_rtp_checkpoint,
    train_rtp_shadow,
)
from poke_bot.recursive_turn_planner.training.losses import compute_rtp_losses
from poke_bot.recursive_turn_planner.training.shadow_train import (
    RTPTrainConfig,
    split_batches_by_game,
    trusted_candidate_targets_from_record,
)
from poke_bot.recursive_turn_planner import (
    PURE_RL_R197_MAX_ACTION_COMBOS,
    RecursiveTurnPlanner,
    get_profile,
    required_recursive_passes,
)
from poke_bot import rtp_evaluation_promotion as promotion


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def _seal_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)
    return path


def _rewrite_sealed_json(path: Path, payload: object) -> Path:
    os.chmod(path, 0o644)
    return _seal_json(path, payload)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _masked_evaluation_receipt(evidence_dir: Path) -> Path:
    """Create physical source evidence which truthfully reaches r197's hold."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[1]
    registry = evidence_dir / "research_control_registry_v1.json"
    registry.write_bytes(
        (repository_root / "ops" / "research_control_registry_v1.json").read_bytes()
    )
    os.chmod(registry, 0o444)
    results = _seal_json(evidence_dir / "evaluation-results.json", {"rows": []})
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
    return _seal_json(evidence_dir / "rtp-evaluation.json", payload)


def _write_accepted_promotion_receipt(
    sidecar: Path,
    *,
    parent_digest: str,
) -> tuple[Path, str]:
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    planner = load_rtp_checkpoint(sidecar)
    config_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            payload["config"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    evidence_dir = sidecar.parent / f"{sidecar.stem}.promotion-evidence"
    evaluation = _masked_evaluation_receipt(evidence_dir)
    deck = evidence_dir / "deck.csv"
    deck.write_text("".join(f"{card}\n" for card in range(1, 61)), encoding="utf-8")
    tree = evidence_dir / "matchup_tree.json"
    tree.write_text('{"schema":"test-r198-tree/v1"}\n', encoding="utf-8")
    receipt = {
        "schema": "poke_bot.rtp_promotion/v1",
        "status": "accepted",
        "specialist_id": "alakazam",
        "parent_checkpoint_sha256": parent_digest,
        "sidecar_sha256": _sha256_file(sidecar),
        "sidecar_config_sha256": config_digest,
        "max_neural_passes": planner.config.max_neural_passes,
        "required_neural_passes": {
            "normal": required_recursive_passes(planner.config),
            "forced_replan": required_recursive_passes(
                planner.config,
                force_recurse=True,
            ),
        },
        "deck_file_sha256": _sha256_file(deck),
        "deck_cards_sha256": promotion.canonical_digest(list(range(1, 61))),
        "matchup_tree_sha256": _sha256_file(tree),
        "evaluation_receipt_path": str(evaluation),
        "evaluation_receipt_sha256": _sha256_file(evaluation),
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
    if planner.config.sizing_profile == "pure_rl_r197":
        receipt["max_action_combos"] = PURE_RL_R197_MAX_ACTION_COMBOS
    path = sidecar.with_suffix(sidecar.suffix + ".promotion.json")
    _seal_json(path, receipt)
    return path, _sha256_file(path)


@pytest.mark.unit
def test_rtp_synthetic_train_and_reload(tmp_path: Path) -> None:
    batches = make_synthetic_batches(n_decisions=24, d_model=96, seed=7)
    result = train_rtp_shadow(
        batches,
        output_dir=tmp_path / "rtp",
        config=RTPTrainConfig(d_model=96, epochs=1, lr=1e-2, seed=7),
    )
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.receipt_path).is_file()
    assert result.metrics["mean_loss"] >= 0.0
    planner = load_rtp_checkpoint(result.checkpoint_path)
    assert int(planner.config.d_model) == 96
    # Forward scores remain finite after load.
    state = batches[0].state
    scores, _ = __import__(
        "poke_bot.recursive_turn_planner.training.shadow_train",
        fromlist=["_action_scores_with_grad"],
    )._action_scores_with_grad(
        planner, state, batches[0].option_hidden, batches[0].legal_actions
    )
    assert torch.isfinite(scores).all()


@pytest.mark.unit
def test_serving_checkpoint_requires_bound_parent_flags_config_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = "sha256:" + "a" * 64
    r197_config = get_profile("pure_rl_r197").to_config()
    r197_path = tmp_path / "r197.pt"
    save_rtp_checkpoint(
        RecursiveTurnPlanner(r197_config),
        r197_path,
        parent_checkpoint_sha256=parent,
        shadow_only=True,
    )
    promotion_path, promotion_digest = _write_accepted_promotion_receipt(
        r197_path,
        parent_digest=parent,
    )
    saved_payload = torch.load(r197_path, map_location="cpu", weights_only=True)
    assert isinstance(saved_payload, dict)
    assert saved_payload["shadow_only"] is True
    assert saved_payload["serving_eligible"] is False
    assert saved_payload["action_authority_enabled"] is False

    with pytest.raises(
        ValueError,
        match="trusted counterfactual candidate targets are absent",
    ):
        load_rtp_checkpoint(
            r197_path,
            expected_parent_digest=parent,
            expected_config=r197_config,
            promotion_receipt=promotion_path,
            expected_promotion_receipt_digest=promotion_digest,
            serving_qualified=True,
        )

    # An arbitrary environment claim must not turn the local source consumer
    # into packaged mode.  Only submission/main.py can first arm that
    # capability after sealing the package pair in-process.
    promotion_payload = json.loads(promotion_path.read_text(encoding="utf-8"))
    with monkeypatch.context() as packaged_env:
        packaged_env.setenv(
            "POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT",
            str(promotion_payload["evaluation_receipt_path"]),
        )
        packaged_env.setenv("POKEBOT_RTP_PROMOTION_RECEIPT", str(promotion_path))
        packaged_env.setenv(
            "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256", promotion_digest
        )
        with pytest.raises(
            ValueError,
            match="trusted counterfactual candidate targets are absent",
        ):
            load_rtp_checkpoint(
                r197_path,
                expected_parent_digest=parent,
                expected_config=r197_config,
                promotion_receipt=promotion_path,
                expected_promotion_receipt_digest=promotion_digest,
                serving_qualified=True,
            )

    invalid_action_receipt = json.loads(promotion_path.read_text(encoding="utf-8"))
    invalid_action_receipt["max_action_combos"] = 256
    invalid_action_path = tmp_path / "invalid-action-cap.promotion.json"
    _seal_json(invalid_action_path, invalid_action_receipt)
    with pytest.raises(
        ValueError,
        match="trusted counterfactual candidate targets are absent",
    ):
        load_rtp_checkpoint(
            r197_path,
            expected_parent_digest=parent,
            expected_config=r197_config,
            promotion_receipt=invalid_action_path,
            expected_promotion_receipt_digest=_sha256_file(invalid_action_path),
            serving_qualified=True,
        )

    with pytest.raises(ValueError, match="granted only by an immutable promotion receipt"):
        save_rtp_checkpoint(
            RecursiveTurnPlanner(r197_config),
            tmp_path / "self_authorized.pt",
            parent_checkpoint_sha256=parent,
            shadow_only=True,
            serving_eligible=True,
        )

    with pytest.raises(ValueError, match="parent digest mismatch"):
        load_rtp_checkpoint(
            r197_path,
            expected_parent_digest="sha256:" + "b" * 64,
            expected_config=r197_config,
            promotion_receipt=promotion_path,
            expected_promotion_receipt_digest=promotion_digest,
            serving_qualified=True,
        )
    with pytest.raises(ValueError, match="serving config mismatch"):
        load_rtp_checkpoint(
            r197_path,
            expected_parent_digest=parent,
            expected_config=get_profile("pure_rl_r197").to_config(repair_budget=2),
            promotion_receipt=promotion_path,
            expected_promotion_receipt_digest=promotion_digest,
            serving_qualified=True,
        )

    # 256 is the global ceiling, so use an in-range but noncanonical profile
    # to exercise the r197 exact-profile gate without constructing an invalid
    # planner config.
    r197_noncanonical_config = get_profile("pure_rl_r197").to_config(
        max_neural_passes=255
    )
    r197_over_budget_path = tmp_path / "r197-noncanonical.pt"
    save_rtp_checkpoint(
        RecursiveTurnPlanner(r197_noncanonical_config),
        r197_over_budget_path,
        parent_checkpoint_sha256=parent,
        shadow_only=True,
    )
    over_budget_promotion_path, over_budget_promotion_digest = (
        _write_accepted_promotion_receipt(
            r197_over_budget_path,
            parent_digest=parent,
        )
    )
    with pytest.raises(ValueError, match="authorized 256-pass profile"):
        load_rtp_checkpoint(
            r197_over_budget_path,
            expected_parent_digest=parent,
            expected_config=r197_noncanonical_config,
            promotion_receipt=over_budget_promotion_path,
            expected_promotion_receipt_digest=over_budget_promotion_digest,
            serving_qualified=True,
        )

    legacy_config = get_profile("pure_rl").to_config()
    legacy_path = tmp_path / "legacy.pt"
    save_rtp_checkpoint(
        RecursiveTurnPlanner(legacy_config),
        legacy_path,
        parent_checkpoint_sha256=parent,
        shadow_only=True,
    )
    legacy_promotion_path, legacy_promotion_digest = _write_accepted_promotion_receipt(
        legacy_path,
        parent_digest=parent,
    )
    # Historical/research loading remains valid, but the impossible 4/2/4
    # recursive path is never eligible for serving action authority.
    assert load_rtp_checkpoint(legacy_path).config == legacy_config
    with pytest.raises(ValueError, match="cannot complete a recursive plan"):
        load_rtp_checkpoint(
            legacy_path,
            expected_parent_digest=parent,
            expected_config=legacy_config,
            promotion_receipt=legacy_promotion_path,
            expected_promotion_receipt_digest=legacy_promotion_digest,
            serving_qualified=True,
        )

    shadow_path = tmp_path / "unpromoted.pt"
    save_rtp_checkpoint(
        RecursiveTurnPlanner(r197_config),
        shadow_path,
        parent_checkpoint_sha256=parent,
        shadow_only=True,
    )
    with pytest.raises(ValueError, match="requires a promotion receipt"):
        load_rtp_checkpoint(
            shadow_path,
            expected_parent_digest=parent,
            expected_config=r197_config,
            serving_qualified=True,
        )

    invalid_receipt = json.loads(promotion_path.read_text(encoding="utf-8"))
    invalid_receipt["action_authority_enabled"] = False
    _rewrite_sealed_json(promotion_path, invalid_receipt)
    with pytest.raises(
        ValueError,
        match="trusted counterfactual candidate targets are absent",
    ):
        load_rtp_checkpoint(
            r197_path,
            expected_parent_digest=parent,
            expected_config=r197_config,
            promotion_receipt=promotion_path,
            expected_promotion_receipt_digest=_sha256_file(promotion_path),
            serving_qualified=True,
        )


@pytest.mark.unit
def test_checkpoint_load_is_safe_and_strict_about_state_dict(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "strict.pt"
    config = get_profile("pure_rl_r197").to_config()
    save_rtp_checkpoint(
        RecursiveTurnPlanner(config),
        checkpoint_path,
        parent_checkpoint_sha256="sha256:" + "a" * 64,
        shadow_only=True,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    state = dict(payload["state_dict"])
    state.pop(next(iter(state)))
    payload["state_dict"] = state
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="state_dict is incompatible"):
        load_rtp_checkpoint(checkpoint_path)


@pytest.mark.unit
def test_ordinary_load_preserves_legacy_partial_r195_style_sidecars(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "r195-style.pt"
    legacy_config = get_profile("pure_rl").to_config()
    save_rtp_checkpoint(RecursiveTurnPlanner(legacy_config), legacy_path)
    payload = torch.load(legacy_path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    # Match the former compact serialization: only fields which can affect
    # module shape/runtime behavior were persisted, and no parent binding was
    # required for ordinary research loading.
    payload["config"] = {
        name: payload["config"][name]
        for name in (
            "sizing_profile",
            "d_model",
            "dynamics_width",
            "num_plan_candidates",
            "max_recursion_depth",
            "max_neural_passes",
            "max_plan_length",
            "complexity_option_threshold",
            "complexity_entropy_threshold",
            "prefer_option_hidden",
            "online_sim_verify_budget",
        )
    }
    payload.pop("parent_checkpoint_sha256")
    payload.pop("shadow_only")
    payload["planner_state_dict"] = payload.pop("state_dict")
    torch.save(payload, legacy_path)

    loaded = load_rtp_checkpoint(legacy_path)
    assert loaded.config.d_model == 96
    assert loaded.config.num_plan_candidates == 4
    assert loaded.config.max_neural_passes == 4
    assert loaded.config.option_batch_hint == 64


@pytest.mark.unit
def test_poke_rlm_shadow_train(tmp_path: Path) -> None:
    batches = make_synthetic_batches(n_decisions=16, d_model=96, seed=3)
    result = train_poke_rlm_shadow(
        batches,
        output_dir=tmp_path / "poke",
        config=PokeRLMTrainConfig(d_model=96, epochs=1, seed=3),
    )
    assert Path(result.checkpoint_path).is_file()
    core = load_poke_rlm_core(result.checkpoint_path)
    assert isinstance(core, PokeRLMModelCore)
    state = batches[0].state.unsqueeze(0)
    opts = batches[0].option_hidden.unsqueeze(0)
    heads = core.score_actions(state, opts)
    route = core.route_logits(state)
    assert heads.policy_logits.shape[-1] == opts.size(1)
    assert route.shape[-1] == 3


@pytest.mark.unit
def test_route_label_root_alias() -> None:
    labels = PlanSupervisionLabels(
        chosen_action_index=0,
        route_target="root",
        should_recurse=False,
        stop_reason="ok",
    )
    bundle = compute_poke_rlm_losses(
        action_logits=torch.randn(4),
        route_logits=torch.randn(3),
        recurse_logits=torch.randn(1),
        labels=labels,
    )
    assert torch.isfinite(bundle.total)


@pytest.mark.unit
def test_train_cli_synthetic(tmp_path: Path) -> None:
    import subprocess
    import sys

    out = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "train_recursive_turn_planner.py"),
            "--out-dir",
            str(out),
            "--synthetic",
            "--n-synthetic",
            "12",
            "--epochs",
            "1",
            "--also-poke-rlm",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source"] == "synthetic"
    assert Path(payload["rtp_checkpoint"]).is_file()
    assert Path(payload["poke_rlm_checkpoint"]).is_file()
    assert (out / "experimental" / "pipeline_summary.json").is_file()


@pytest.mark.unit
def test_rtp_game_level_split_and_shadow_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="global authorized ceiling"):
        RTPTrainConfig(max_neural_passes=257)

    batches = make_synthetic_batches(n_decisions=16, d_model=96, seed=11)
    train, heldout, split = split_batches_by_game(
        batches, heldout_fraction=0.25, seed=19
    )
    assert split["n_games"] == 4
    assert split["n_train_games"] == 3
    assert split["n_heldout_games"] == 1
    assert {batch.episode_id for batch in train}.isdisjoint(
        {batch.episode_id for batch in heldout}
    )

    result = train_rtp_shadow(
        train,
        heldout_batches=heldout,
        output_dir=tmp_path / "rtp",
        config=RTPTrainConfig(
            d_model=96,
            profile="pure_rl_r197",
            max_neural_passes=256,
            epochs=1,
            seed=11,
        ),
        provenance={
            "game_heldout_split": split,
            "parent_digest": "sha256:" + "a" * 64,
        },
        parent_checkpoint_sha256="sha256:" + "a" * 64,
    )
    assert result.heldout_metrics["available"] is True
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["shadow_only"] is True
    assert payload["research_only"] is False
    assert payload["serving_eligible"] is False
    assert payload["action_authority_enabled"] is False
    assert payload["parent_checkpoint_sha256"] == "sha256:" + "a" * 64
    assert payload["config"]["max_neural_passes"] == 256
    assert payload["extra"]["required_recursive_passes"]["normal_recursive"] == 6
    assert payload["extra"]["provenance"]["game_heldout_split"] == split


@pytest.mark.unit
def test_rtp_selected_outcome_losses_never_relabel_unchosen_actions() -> None:
    scores = torch.tensor([0.2, -0.4], requires_grad=True)
    value = torch.tensor([0.1], requires_grad=True)
    uncertainty = torch.tensor([0.3], requires_grad=True)
    masked = compute_rtp_losses(
        action_scores=scores,
        chosen_action_index=99,
        complexity_logit=torch.tensor(0.0),
        chosen_value_prediction=value,
        chosen_uncertainty=uncertainty,
        game_value=1.0,
    )
    assert masked.metadata["action_target_available"] is False
    assert masked.metadata["value_target_available"] is False
    assert masked.action.item() == 0.0
    assert masked.value.item() == 0.0

    observed = compute_rtp_losses(
        action_scores=scores,
        chosen_action_index=0,
        complexity_logit=torch.tensor(0.0),
        chosen_value_prediction=value,
        chosen_uncertainty=uncertainty,
        game_value=1.0,
    )
    assert observed.metadata["value_target_available"] is True
    assert observed.metadata["calibration_target_available"] is True
    assert observed.ranking.item() > 0.0


@pytest.mark.unit
def test_candidate_evaluator_targets_require_a_trusted_action_space_binding() -> None:
    fingerprint = "sha256:" + "f" * 64
    absent = trusted_candidate_targets_from_record(
        {}, n_actions=2, action_space_fingerprint=fingerprint
    )
    assert absent["candidate_return_targets"] is None
    assert absent["provenance"]["status"] == "not_supplied"

    untrusted = trusted_candidate_targets_from_record(
        {
            "evaluator_targets": {
                "schema": "poke_bot.rtp_complete_action_evaluator_targets/v1",
                "trusted": False,
                "action_space_fingerprint": fingerprint,
                "evaluator_receipt_sha256": "sha256:" + "a" * 64,
                "candidate_return_targets": [1.0, -1.0],
            }
        },
        n_actions=2,
        action_space_fingerprint=fingerprint,
    )
    assert untrusted["candidate_return_targets"] is None
    assert untrusted["provenance"]["status"] == "masked_untrusted_evaluator_targets"

    trusted = trusted_candidate_targets_from_record(
        {
            "evaluator_targets": {
                "schema": "poke_bot.rtp_complete_action_evaluator_targets/v1",
                "trusted": True,
                "action_space_fingerprint": fingerprint,
                "evaluator_receipt_sha256": "sha256:" + "a" * 64,
                "candidate_return_targets": [1.0, -1.0],
                "candidate_ranking_targets": [0.8, 0.2],
                "candidate_calibration_targets": [0.1, 0.4],
            }
        },
        n_actions=2,
        action_space_fingerprint=fingerprint,
    )
    assert trusted["provenance"]["status"] == "trusted_action_space_bound"
    assert trusted["candidate_return_mask"].tolist() == [True, True]
    assert trusted["candidate_ranking_mask"].tolist() == [True, True]
    assert trusted["candidate_calibration_mask"].tolist() == [True, True]
