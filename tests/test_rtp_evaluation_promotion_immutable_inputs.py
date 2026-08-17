"""Fail-closed source-evidence coverage for the r198 promotion consumer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from poke_bot import rtp_evaluation_promotion as promotion


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _digest(path), "bytes": path.stat().st_size}


def _write_json(path: Path, payload: object, *, immutable: bool) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        os.chmod(path, 0o444)
    return path


def _receipt(
    tmp_path: Path,
    *,
    results_path: Path | None = None,
    result_in_memory: bool = False,
    status: str = "ready_for_separate_promotion_review",
) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "research_control_registry.json"
    registry.write_bytes((ROOT / "ops/research_control_registry_v1.json").read_bytes())
    os.chmod(registry, 0o444)
    if results_path is None:
        results_path = _write_json(tmp_path / "results.json", {"rows": []}, immutable=True)
    value: dict[str, object] = {
        "schema": promotion.EVALUATION_RECEIPT_SCHEMA,
        "status": status,
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
        "results": {**_identity(results_path), "in_memory": result_in_memory},
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
    material = {
        key: item
        for key, item in value.items()
        if key not in {"created_at_utc", "receipt_input_sha256"}
    }
    value["receipt_input_sha256"] = promotion.canonical_digest(material)
    receipt = _write_json(tmp_path / "evaluation.json", value, immutable=True)
    return receipt, value


def _forced_turn_order_trace() -> list[dict[str, Any]]:
    return [
        {
            "control_index": 0,
            "control": "forced_go_first_contract",
            "prompt_context": 41,
            "prompt_context_encoding": "numeric_41",
            "expected_action": [0],
            "returned_action": [0],
            "verified_observation_action_contract": True,
            "rtp_diagnostics_absent": True,
            "complexity_probe_not_invoked": True,
            "excluded_from_candidate_decisions": True,
            "excluded_from_intended_complex_denominator": True,
            "excluded_from_latency": True,
        }
    ]


def _telemetry(*, candidate_seat: int) -> dict[str, Any]:
    if candidate_seat not in {0, 1}:
        raise AssertionError("fixture has an invalid candidate seat")
    return {
        "candidate_decisions": 1,
        "planner_eligible_candidate_decisions": 1,
        "over_cap_factorized_fallback_decisions": 0,
        "over_cap_factorized_fallback_trace": [],
        "over_cap_factorized_fallback_trace_sha256": promotion.canonical_digest([]),
        "forced_turn_order_controls": int(candidate_seat == 0),
        "forced_turn_order_control_trace": (
            _forced_turn_order_trace() if candidate_seat == 0 else []
        ),
        "intended_complex_decisions": 0,
        "recursive_intended_complex_decisions": 0,
        "successful_recursive_intended_complex_decisions": 0,
        "direct_bridge_decisions": 0,
        "recursive_decisions": 0,
        "fallback_decisions": 0,
        "unexpected_recursive_fallback_decisions": 0,
        "expected_recursive_fallback_decisions": 0,
        "neural_budget_exceeded": 0,
        "neural_budget_failures": 0,
        "illegal_action_count": 0,
        "candidate_forfeit_count": 0,
        "latency_seconds": 0.01,
        "intended_complex_decision_scope": "new_turn_complexity_gate_only",
        "recursive_mode_counts": {
            "continue_plan": 0,
            "direct_policy_fallback": 0,
            "recursive_plan": 0,
            "replan_direct": 0,
            "replan_with_program": 0,
        },
        "decision_latency_trace": [
            {
                "decision_index": 0,
                "mode": "no_rtp",
                "planner_mode": "no_rtp",
                "planner_reason": "baseline",
                "intended_complex": False,
                "fallback_classification": None,
                "latency_seconds": 0.01,
            }
        ],
        "recursive_decision_latency_seconds": [],
        "normal_recursive_plan_passes": [],
        "forced_replan_passes": [],
    }


def _over_cap_telemetry(*, arm: str, candidate_seat: int) -> dict[str, Any]:
    telemetry = _telemetry(candidate_seat=candidate_seat)
    action_space = {
        "n_options": 9,
        "min_count": 1,
        "max_count": 5,
        "counts": [1, 2, 3, 4, 5],
        "complete_ordered_action_cardinality": 18_729,
        "complete_ordered_action_cap": 1024,
        "over_cap": True,
        "complete_ordered_actions_materialized": False,
        "complete_ordered_action_truncated": False,
    }
    action_space_sha256 = promotion.canonical_digest(action_space)
    observation_sha256 = "sha256:" + "b" * 64
    policy_input_sha256 = "sha256:" + "c" * 64
    logical_pre_action_sha256 = promotion.canonical_digest(
        {
            "observation_sha256": observation_sha256,
            "action_space_sha256": action_space_sha256,
            "candidate_policy_input_sha256": policy_input_sha256,
        }
    )
    diagnostic: dict[str, Any] | None = None
    if arm != "no_rtp":
        diagnostic = {
            "mode": "fallback",
            "fallback_code": "action_space_too_large",
            "neural_passes": 0,
            "required_neural_passes": 0,
            "legal_count": 0,
            "decision_mode": "",
        }
    trace = {
        "decision_index": 0,
        "arm": arm,
        "mode": "over_cap_factorized_fallback",
        "classification": "complete_ordered_action_space_over_cap",
        "action_space": action_space,
        "action_space_sha256": action_space_sha256,
        "observation_sha256": observation_sha256,
        "candidate_policy_input_sha256": policy_input_sha256,
        "logical_pre_action_sha256": logical_pre_action_sha256,
        "returned_action": [0, 2, 3, 6, 5],
        "factorized_teacher_forcing_legal": True,
        "factorized_teacher_forcing_stage_count": 5,
        "complexity_probe_not_invoked": True,
        "neural_passes": 0,
        "required_neural_passes": 0,
        "neural_budget_failure": False,
        "rtp_diagnostic": diagnostic,
        "included_in_candidate_decisions": True,
        "included_in_candidate_latency": True,
        "excluded_from_planner_eligible_candidate_decisions": True,
        "excluded_from_intended_complex_denominator": True,
        "excluded_from_direct_bridge_metrics": True,
        "excluded_from_recursive_metrics": True,
        "excluded_from_fallback_metrics": True,
        "excluded_from_neural_pass_metrics": True,
        "excluded_from_recursive_latency": True,
    }
    telemetry.update(
        {
            "planner_eligible_candidate_decisions": 0,
            "over_cap_factorized_fallback_decisions": 1,
            "over_cap_factorized_fallback_trace": [trace],
            "over_cap_factorized_fallback_trace_sha256": promotion.canonical_digest([trace]),
            "intended_complex_decisions": 0,
            "recursive_intended_complex_decisions": 0,
            "successful_recursive_intended_complex_decisions": 0,
            "direct_bridge_decisions": 0,
            "recursive_decisions": 0,
            "fallback_decisions": 0,
            "unexpected_recursive_fallback_decisions": 0,
            "expected_recursive_fallback_decisions": 0,
            "normal_recursive_plan_passes": [],
            "forced_replan_passes": [],
            "recursive_mode_counts": {
                "continue_plan": 0,
                "direct_policy_fallback": 0,
                "recursive_plan": 0,
                "replan_direct": 0,
                "replan_with_program": 0,
            },
            "latency_seconds": 0.04,
            "decision_latency_trace": [
                {
                    "decision_index": 0,
                    "mode": "over_cap_factorized_fallback",
                    "planner_mode": "over_cap_factorized_fallback",
                    "planner_reason": "complete_ordered_action_space_over_cap",
                    "intended_complex": None,
                    "fallback_classification": None,
                    "latency_seconds": 0.04,
                    "over_cap_trace_index": 0,
                }
            ],
            "recursive_decision_latency_seconds": [],
        }
    )
    return telemetry


@pytest.mark.unit
@pytest.mark.parametrize("arm", promotion.ARMS)
def test_promotion_recomputes_over_cap_telemetry_and_rate_denominator(arm: str) -> None:
    raw = _over_cap_telemetry(arm=arm, candidate_seat=1)
    telemetry = promotion._validated_row_telemetry(raw, arm=arm, label="over cap")
    assert telemetry["candidate_decisions"] == 1
    assert telemetry["planner_eligible_candidate_decisions"] == 0
    assert telemetry["over_cap_factorized_fallback_decisions"] == 1
    summary = promotion._recompute_arm_summary(
        [
            {
                "arm": arm,
                "candidate_score": 1.0,
                "opponent_id": "iono",
                "candidate_seat": seat,
                "telemetry": _over_cap_telemetry(arm=arm, candidate_seat=seat),
            }
            for seat in ([0] * 500 + [1] * 500)
        ]
    )
    assert summary["telemetry"]["fallback_rate"] is None
    assert summary["telemetry"]["over_cap_factorized_fallback_rate"] == 1.0
    assert summary["telemetry"]["over_cap_factorized_fallback_trace"]["count"] == 1_000

    coerced = json.loads(json.dumps(raw))
    coerced["over_cap_factorized_fallback_trace"][0]["action_space"]["max_count"] = "5"
    coerced["over_cap_factorized_fallback_trace_sha256"] = promotion.canonical_digest(
        coerced["over_cap_factorized_fallback_trace"]
    )
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="exact non-bool integer"):
        promotion._validated_row_telemetry(coerced, arm=arm, label="coerced")

    tampered_diagnostic = json.loads(json.dumps(raw))
    if arm == "no_rtp":
        tampered_diagnostic["over_cap_factorized_fallback_trace"][0]["rtp_diagnostic"] = {}
    else:
        tampered_diagnostic["over_cap_factorized_fallback_trace"][0]["rtp_diagnostic"][
            "required_neural_passes"
        ] = 6
    tampered_diagnostic["over_cap_factorized_fallback_trace_sha256"] = promotion.canonical_digest(
        tampered_diagnostic["over_cap_factorized_fallback_trace"]
    )
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="diagnostic"):
        promotion._validated_row_telemetry(
            tampered_diagnostic, arm=arm, label="diagnostic tamper"
        )


@pytest.mark.unit
def test_promotion_over_cap_action_parity_is_only_for_equal_logical_inputs() -> None:
    same = {
        "cell-000000": {
            arm: promotion._validated_row_telemetry(
                _over_cap_telemetry(arm=arm, candidate_seat=1),
                arm=arm,
                label=f"{arm} special",
            )["over_cap_factorized_fallback_trace"]
            for arm in promotion.ARMS
        }
    }
    observed = promotion._validate_conditional_over_cap_action_parity(same)
    assert observed["cross_arm_comparable_groups"] == 1
    assert observed["cross_arm_comparable_arm_rows"] == 3

    divergent = json.loads(json.dumps(same))
    for index, arm in enumerate(promotion.ARMS):
        trace = divergent["cell-000000"][arm][0]
        trace["candidate_policy_input_sha256"] = "sha256:" + str(index + 1) * 64
        trace["logical_pre_action_sha256"] = promotion.canonical_digest(
            {
                "observation_sha256": trace["observation_sha256"],
                "action_space_sha256": trace["action_space_sha256"],
                "candidate_policy_input_sha256": trace["candidate_policy_input_sha256"],
            }
        )
        trace["returned_action"] = [index]
    assert promotion._validate_conditional_over_cap_action_parity(divergent)[
        "cross_arm_comparable_groups"
    ] == 0

    mismatch = json.loads(json.dumps(same))
    mismatch["cell-000000"][promotion.DIRECT_BRIDGE_ARM][0]["returned_action"] = [
        1,
        2,
        3,
        4,
        5,
    ]
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="actions differ"):
        promotion._validate_conditional_over_cap_action_parity(mismatch)


@pytest.mark.unit
def test_promotion_revalidates_exact_forced_turn_order_controls() -> None:
    raw = _telemetry(candidate_seat=0)
    telemetry = promotion._validated_row_telemetry(
        raw, arm="no_rtp", label="forced control"
    )
    promotion._validate_r198_forced_turn_order_contract(
        telemetry, candidate_seat=0, label="forced control"
    )
    assert telemetry["forced_turn_order_control_trace"] == _forced_turn_order_trace()

    wrong_action = json.loads(json.dumps(raw))
    wrong_action["forced_turn_order_control_trace"][0]["returned_action"] = [1]
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="differs"):
        promotion._validated_row_telemetry(
            wrong_action, arm="no_rtp", label="wrong forced action"
        )

    float_context = json.loads(json.dumps(raw))
    float_context["forced_turn_order_control_trace"][0]["prompt_context"] = 41.0
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="invalid prompt context"):
        promotion._validated_row_telemetry(
            float_context, arm="no_rtp", label="float context"
        )

    float_control_index = json.loads(json.dumps(raw))
    float_control_index["forced_turn_order_control_trace"][0]["control_index"] = 0.0
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="indexes are not monotonic"):
        promotion._validated_row_telemetry(
            float_control_index, arm="no_rtp", label="float control index"
        )

    float_action = json.loads(json.dumps(raw))
    float_action["forced_turn_order_control_trace"][0]["expected_action"] = [0.0]
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="action index"):
        promotion._validated_row_telemetry(
            float_action, arm="no_rtp", label="float action"
        )

    string_action = json.loads(json.dumps(raw))
    string_action["forced_turn_order_control_trace"][0]["returned_action"] = ["0"]
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="action index"):
        promotion._validated_row_telemetry(
            string_action, arm="no_rtp", label="string action"
        )

    diagnostics_tamper = json.loads(json.dumps(raw))
    diagnostics_tamper["forced_turn_order_control_trace"][0][
        "rtp_diagnostics_absent"
    ] = False
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="rtp_diagnostics_absent"):
        promotion._validated_row_telemetry(
            diagnostics_tamper, arm="no_rtp", label="diagnostics tamper"
        )

    diagnostic_payload_tamper = json.loads(json.dumps(raw))
    diagnostic_payload_tamper["forced_turn_order_control_trace"][0]["diagnostic"] = {
        "mode": "direct_bridge"
    }
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="exact canonical field set"):
        promotion._validated_row_telemetry(
            diagnostic_payload_tamper, arm="no_rtp", label="diagnostic payload tamper"
        )

    latency_tamper = json.loads(json.dumps(raw))
    latency_tamper["latency_seconds"] = 0.02
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="does not equal"):
        promotion._validated_row_telemetry(
            latency_tamper, arm="no_rtp", label="latency tamper"
        )

    no_control = promotion._validated_row_telemetry(
        _telemetry(candidate_seat=1), arm="no_rtp", label="seat one"
    )
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="frozen seat ABI"):
        promotion._validate_r198_forced_turn_order_contract(
            no_control, candidate_seat=0, label="missing seat zero control"
        )

    paired = {
        "cell-000000": {
            arm: {"count": 1, "trace": _forced_turn_order_trace()}
            for arm in promotion.ARMS
        }
    }
    promotion._verify_forced_turn_order_pairing(paired)
    paired["cell-000000"][promotion.DIRECT_BRIDGE_ARM]["trace"][0][
        "expected_action"
    ] = [1]
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="traces differ"):
        promotion._verify_forced_turn_order_pairing(paired)

    rows = [
        {
            "arm": "no_rtp",
            "candidate_score": 1.0,
            "opponent_id": "iono",
            "candidate_seat": seat,
            "telemetry": _telemetry(candidate_seat=seat),
        }
        for seat in ([0] * 500 + [1] * 500)
    ]
    summary = promotion._recompute_arm_summary(rows)
    assert summary["telemetry"]["forced_turn_order_controls"] == 500
    assert summary["telemetry"]["forced_turn_order_control_trace"]["count"] == 500


@pytest.mark.unit
def test_source_results_must_be_physical_exact_0444(tmp_path: Path) -> None:
    results = _write_json(tmp_path / "writable-results.json", {"rows": []}, immutable=False)
    receipt, _ = _receipt(tmp_path, results_path=results)

    with pytest.raises(promotion.RTPPromotionEvidenceError, match="mode 0444"):
        promotion.validate_r198_evaluation_receipt(receipt)


@pytest.mark.unit
def test_source_results_and_receipt_symlinks_are_rejected(tmp_path: Path) -> None:
    source_results = _write_json(tmp_path / "source-results.json", {"rows": []}, immutable=True)
    result_link = tmp_path / "results-link.json"
    result_link.symlink_to(source_results)
    receipt, _ = _receipt(tmp_path, results_path=result_link)

    with pytest.raises(promotion.RTPPromotionEvidenceError, match="symbolic link"):
        promotion.validate_r198_evaluation_receipt(receipt)

    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(receipt)
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="symbolic link"):
        promotion.validate_r198_evaluation_receipt(receipt_link)


@pytest.mark.unit
def test_hold_and_raw_evaluation_receipts_never_reach_promotion(tmp_path: Path) -> None:
    hold, _ = _receipt(tmp_path / "hold", status="hold")
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="not ready"):
        promotion.validate_r198_evaluation_receipt(hold)

    raw, _ = _receipt(tmp_path / "raw", result_in_memory=True)
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="in-memory"):
        promotion.validate_r198_evaluation_receipt(raw)


@pytest.mark.unit
@pytest.mark.parametrize(
    "status",
    ["failed_closed_nonresult", "abandoned", "continued"],
)
def test_r199_continuation_words_never_broaden_promotion_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    receipt, _ = _receipt(tmp_path / status, status=status)

    with pytest.raises(promotion.RTPPromotionEvidenceError, match="not ready"):
        promotion.validate_r198_evaluation_receipt(receipt)


@pytest.mark.unit
def test_tampered_evaluation_receipt_digest_is_rejected_before_claims(tmp_path: Path) -> None:
    receipt, original = _receipt(tmp_path)
    expected = _digest(receipt)
    os.chmod(receipt, 0o644)
    original["status"] = "hold"
    receipt.write_text(json.dumps(original, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(receipt, 0o444)

    with pytest.raises(promotion.RTPPromotionEvidenceError, match="digest mismatch"):
        promotion.validate_r198_evaluation_receipt(receipt, expected_sha256=expected)


@pytest.mark.unit
def test_fixed_current_masked_candidate_is_a_hard_promotion_hold(tmp_path: Path) -> None:
    receipt, _ = _receipt(tmp_path)

    with pytest.raises(
        promotion.RTPPromotionEvidenceError,
        match="trusted counterfactual candidate targets are absent",
    ):
        promotion.validate_r198_evaluation_receipt(receipt)


@pytest.mark.unit
def test_full_hold_audits_rows_before_fixed_masked_target_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known HOLD cannot hide malformed over-cap evidence from audit."""

    receipt, payload = _receipt(tmp_path)
    os.chmod(receipt, 0o644)
    payload["status"] = "hold"
    payload["validated_rows"] = []  # marks this as a full-shaped audit path
    payload["promotion_decision"] = {
        "eligible_for_separate_promotion_review": False,
        "self_promotion_performed": False,
        "serving_change_authorized": False,
    }
    material = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "receipt_input_sha256"}
    }
    payload["receipt_input_sha256"] = promotion.canonical_digest(material)
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(receipt, 0o444)

    stages: list[str] = []
    monkeypatch.setattr(
        promotion,
        "_validate_panel_and_rows",
        lambda *_args, **_kwargs: stages.append("rows") or [],
    )
    monkeypatch.setattr(
        promotion,
        "_validate_efficacy_semantics",
        lambda *_args, **_kwargs: stages.append("efficacy"),
    )
    monkeypatch.setattr(
        promotion,
        "_validate_expected_candidate_bindings",
        lambda *_args, **_kwargs: stages.append("bindings"),
    )
    with pytest.raises(
        promotion.RTPPromotionEvidenceError,
        match="trusted counterfactual candidate targets are absent",
    ):
        promotion.validate_r198_evaluation_receipt(receipt)
    assert stages == ["rows", "efficacy", "bindings"]

    def malformed_rows(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        stages.append("malformed")
        raise promotion.RTPPromotionEvidenceError("over-cap row malformed")

    monkeypatch.setattr(promotion, "_validate_panel_and_rows", malformed_rows)
    with pytest.raises(promotion.RTPPromotionEvidenceError, match="over-cap row malformed"):
        promotion.validate_r198_evaluation_receipt(receipt)


@pytest.mark.unit
def test_packaged_evaluation_env_requires_an_armed_sealed_capability(
    tmp_path: Path,
) -> None:
    evaluation, _ = _receipt(tmp_path / "package")
    packaged_promotion = _write_json(
        tmp_path / "package" / "rtp_promotion_receipt.json",
        {"evaluation_receipt_sha256": _digest(evaluation)},
        immutable=True,
    )
    profile = {
        "rtp_mode": "recursive",
        "rtp_promotion_receipt_file": "rtp_promotion_receipt.json",
        "rtp_promotion_receipt_sha256": _digest(packaged_promotion),
        "rtp_evaluation_receipt_file": "rtp_evaluation_receipt.json",
        "rtp_evaluation_receipt_sha256": _digest(evaluation),
    }
    packaged_evaluation = tmp_path / "package" / "rtp_evaluation_receipt.json"
    packaged_evaluation.write_bytes(evaluation.read_bytes())
    os.chmod(packaged_evaluation, 0o444)
    profile["rtp_evaluation_receipt_sha256"] = _digest(packaged_evaluation)
    os.chmod(packaged_promotion, 0o644)
    packaged_promotion.write_text(
        json.dumps(
            {"evaluation_receipt_sha256": _digest(packaged_evaluation)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(packaged_promotion, 0o444)
    profile["rtp_promotion_receipt_sha256"] = _digest(packaged_promotion)
    environment = {
        "POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT": str(packaged_evaluation),
        "POKEBOT_RTP_PROMOTION_RECEIPT": str(packaged_promotion),
        "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256": _digest(packaged_promotion),
    }

    # A generic environment assignment is only a hint: it must fall back to
    # local source evidence instead of granting the package-only exception.
    assert (
        promotion.resolve_r198_packaged_evaluation_capability(environment=environment)
        is None
    )

    capability = promotion.arm_r198_packaged_evaluation_capability(
        package_root=tmp_path / "package", runtime_profile=profile
    )
    assert promotion.resolve_r198_packaged_evaluation_capability(
        environment=environment
    ) == capability


@pytest.mark.unit
def test_packaged_capability_refuses_noncanonical_receipt_filenames(tmp_path: Path) -> None:
    profile = {
        "rtp_mode": "recursive",
        "rtp_promotion_receipt_file": "other-promotion.json",
        "rtp_evaluation_receipt_file": "rtp_evaluation_receipt.json",
    }

    with pytest.raises(promotion.RTPPromotionEvidenceError, match="cannot arm"):
        promotion.arm_r198_packaged_evaluation_capability(
            package_root=tmp_path, runtime_profile=profile
        )
