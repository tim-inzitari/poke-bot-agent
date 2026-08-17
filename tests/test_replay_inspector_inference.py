from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest
import torch

from poke_bot import config, features
from poke_bot.model import build_model
from replay_inspector import inference
from replay_inspector.parameters import ParameterInspectionError, ParameterInspector


def _stub_feature_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unit fixture independent of an installed competition runtime."""

    monkeypatch.setattr(features, "card_vocab_size", lambda: 8)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 8)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 32)


def _board_tokens() -> features.SparseVector:
    result = features.SparseVector()
    for _ in range(24):
        result.word_start()
        result.add(1, 1.0)
    return result


def _option_tokens(_obs: object, combos: list[list[int]]) -> features.SparseVector:
    result = features.SparseVector()
    for combo in combos:
        result.word_start()
        result.add(1 + sum(combo), 1.0)
    return result


def _h10_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    latent_action_authority: bool = False,
    matchup_adapters_enabled: bool = False,
) -> torch.nn.Module:
    _stub_feature_vocabulary(monkeypatch)
    d_model = 96 if matchup_adapters_enabled else 16
    cfg = config.ModelConfig(
        d_model=d_model,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=max(32, d_model * 2),
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
        expanded_heads_enabled=True,
        setup_board_outcome_head_enabled=True,
        combo_state_head_enabled=True,
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=True,
        decision_fusion_dedicated_routes_enabled=True,
        decision_fusion_dedicated_routes_runtime_enabled=True,
        decision_fusion_typed_output_centered_routes_enabled=True,
        decision_fusion_action_type_reliability_cap=0.25,
        decision_fusion_width=8,
        h10_capacity_enabled=True,
        h10_head_residual_width=8,
        latent_lookahead_enabled=latent_action_authority,
        latent_lookahead_action_authority_enabled=latent_action_authority,
        latent_lookahead_width=8,
        matchup_adapters_enabled=matchup_adapters_enabled,
    )
    model = build_model(
        cfg,
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=8,
    )
    with torch.no_grad():
        assert model.decision_fusion is not None
        for route in model.decision_fusion.dedicated_routes.values():
            route.network[-1].weight.fill_(0.1)
            route.network[-1].bias.fill_(0.01)
    return model.eval()


def _legacy_model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    _stub_feature_vocabulary(monkeypatch)
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
    )
    return build_model(
        cfg,
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=8,
    ).eval()


def _replay() -> dict[str, object]:
    observation = {
        "current": {"turn": 3, "yourIndex": 0},
        "select": {
            "context": 1,
            "option": [{}, {}],
            "minCount": 1,
            "maxCount": 1,
        },
    }
    return {
        "steps": [
            [{"status": "ACTIVE", "observation": observation}, {}],
            [{"action": [1]}, {}],
        ]
    }


def _two_decision_replay() -> dict[str, object]:
    first = {
        "current": {"turn": 3, "yourIndex": 0},
        "select": {"context": 1, "option": [{}, {}]},
    }
    second = {
        "current": {"turn": 4, "yourIndex": 0},
        "select": {"context": 1, "option": [{}, {}]},
    }
    return {
        "steps": [
            [{"status": "ACTIVE", "observation": first}, {}],
            [{"action": [1]}, {}],
            [{"status": "ACTIVE", "observation": second}, {}],
            [{"action": [1]}, {}],
        ]
    }


def _is_first_then_decision_replay() -> dict[str, object]:
    setup = {
        "current": {"turn": 0, "yourIndex": 0},
        "select": {"context": "IsFirst", "option": [{}, {}]},
    }
    decision = {
        "current": {"turn": 1, "yourIndex": 0},
        "select": {"context": 1, "option": [{}, {}]},
    }
    return {
        "steps": [
            [{"status": "ACTIVE", "observation": setup}, {}],
            [{"action": [1]}, {}],
            [{"status": "ACTIVE", "observation": decision}, {}],
            [{"action": [1]}, {}],
        ]
    }


class _Router:
    candidate_model_route = 1

    def __init__(self) -> None:
        self.observed: list[object] = []

    def fork(self) -> _Router:
        return _Router()

    def observe(self, observation: object, **_kwargs: object) -> None:
        self.observed.append(observation)

    def snapshot(self, **_kwargs: object) -> dict[str, object]:
        return {"tree_digest": "sha256:test-router", "observations": len(self.observed)}


class _UnknownRouter(_Router):
    candidate_model_route = -1

    def fork(self) -> _UnknownRouter:
        return _UnknownRouter()


class _CausalRouter(_Router):
    @property
    def candidate_model_route(self) -> int:
        return 1 if len(self.observed) >= 2 else -1

    def fork(self) -> _CausalRouter:
        return _CausalRouter()


def _submitted_runtime_activation() -> dict[str, object]:
    return {
        "basis": "checksum_bound_submitted_startup",
        "runtime_parity_verified": True,
        "runtime_identity_verified": True,
        "runtime_source_tree_sha256": "sha256:" + "a" * 64,
        "matchup_tree_verified": True,
        "matchup_tree_sha256": "sha256:" + "b" * 64,
        "submitted_startup_behavior_verified": True,
        "submitted_startup_behavior": (
            "packaged_matchup_tree_enables_policy_agent_runtime"
        ),
    }


def _materialize_trained_adapter(model: torch.nn.Module, route: int = 1) -> None:
    bank = model.matchup_adapter_bank
    with torch.no_grad():
        bank.experts[route].up.bias.fill_(0.25)
    bank.enabled = False
    bank.dormant_provenance = {
        "schema": "poke_bot.trained_dormant_matchup_adapter/v1",
        "zero_output": False,
    }


@pytest.fixture
def feature_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference.features, "assert_info_set", lambda _obs: None)
    monkeypatch.setattr(
        inference.features, "build_board_tokens", lambda _obs, _deck: _board_tokens()
    )
    monkeypatch.setattr(inference.features, "build_option_tokens", _option_tokens)
    monkeypatch.setattr(
        inference.features,
        "factorized_teacher_forcing_stages",
        lambda _obs, _action: [([[0], [1]], 1)],
    )


def test_h10_replay_trace_exposes_heads_routes_and_exact_loo(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        router=_Router(),
        checkpoint_digest="sha256:test-model",
    )

    assert payload["availability"]["available"] is True
    assert payload["provenance"]["reproduction_status"] == "recomputed_not_historical"
    assert payload["replay"]["router"]["route"] == 1
    assert payload["replay"]["router"]["reconstruction_parity"]["status"] == (
        "recomputed_router_path"
    )
    assert len(payload["policy"]["base_logits"]) == 2
    assert sum(payload["policy"]["probabilities"]) == pytest.approx(1.0)
    assert payload["heads"]["combo_state"]["availability"]["available"] is True
    assert (
        payload["heads"]["combo_state"]["h10_residual"]["availability"]["available"]
        is True
    )
    assert payload["heads"]["action_type"]["normalization"]["kind"] == (
        "softmax_over_legal_candidates"
    )

    route = payload["fusion"]["routes"]["routes"]["action_type"]
    assert len(route["raw_route_delta"]) == 2
    assert route["reliability"]["kind"] == "learned_positive_bounded"
    assert route["reliability"]["effective_multiplier"] == pytest.approx(0.25)
    loo = payload["heads"]["action_type"]["leave_one_out"]
    assert loo["availability"]["available"] is True
    assert len(loo["runtime_path"]["effect_logits"]) == 2
    influence = payload["heads"]["action_type"]["policy_influence"]
    assert influence["availability"]["available"] is True
    assert influence["method"] == (
        "exact_leave_one_head_out_final_policy_recomputation"
    )
    assert influence["sign_convention"] == "full_policy_minus_policy_without_head"
    assert influence["selected_option_index"] == payload["policy"]["model_choice_index"]
    assert influence["maximum_absolute_option_probability_delta"] >= 0.0
    runtime_loo = loo["runtime_path"]
    assert sum(runtime_loo["effect_probabilities"]) == pytest.approx(0.0, abs=1e-6)
    assert influence["selected_option_probability_delta"] == pytest.approx(
        runtime_loo["effect_probabilities"][influence["selected_option_index"]]
    )
    assert 0.0 <= influence["total_variation_distance"] <= 1.0
    adapter = payload["adapter"]
    assert adapter["decision_route_active"] is False
    assert adapter["runtime_enabled"] is False
    assert adapter["policy_influence"]["availability"]["available"] is False
    assert (
        "startup activation evidence"
        in (adapter["policy_influence"]["availability"]["reason"])
    )
    assert (
        payload["heads"]["action_type"]["mask"]["training_target_mask"]["available"]
        is False
    )


@pytest.mark.parametrize(
    ("actor", "missing"),
    [(None, True), (1, False), (True, False), ("0", False)],
)
def test_direct_model_trace_requires_exact_archived_actor_identity(
    monkeypatch: pytest.MonkeyPatch,
    feature_stubs: None,
    actor: object,
    missing: bool,
) -> None:
    replay = _replay()
    current = replay["steps"][0][0]["observation"]["current"]
    if missing:
        del current["yourIndex"]
    else:
        current["yourIndex"] = actor

    payload = inference.inspect_replay_step(
        model=_h10_model(monkeypatch),
        replay=replay,
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
    )

    assert payload["availability"] == {
        "available": False,
        "reason": (
            "active selectable replay row has no exact archived acting-seat identity"
        ),
        "code": "acting_seat_identity_unavailable",
    }
    assert payload["unavailable"]["blocking_step"] == 0


def test_decision_influence_exact_scaling_has_one_x_and_zero_x_parity(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch)
    parameter_before = (
        model.decision_fusion.dedicated_routes["action_type"]
        .network[-1]
        .weight.detach()
        .clone()
    )

    one = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        head_scales={"action_type": 1.0},
    )
    one_influence = one["decision_influence"]
    assert one_influence["availability"]["available"] is True
    assert one_influence["mode"] == "decision_only_counterfactual"
    assert one_influence["method"] == ("exact_nonlinear_fusion_source_recomputation")
    assert one_influence["training_weight"] is False
    assert one_influence["reproduction_status"] == "recomputed_not_historical"
    assert one_influence["parity"]["one_x_matches_runtime_baseline"] is True
    action_type_weight = one_influence["baseline_head_weights"]["action_type"]
    reliability = one["fusion"]["routes"]["routes"]["action_type"]["reliability"][
        "effective_multiplier"
    ]
    assert action_type_weight["source_scale"] == 1.0
    assert action_type_weight["learned_route_multiplier"] == pytest.approx(reliability)
    assert action_type_weight["nominal_policy_coefficient"] == pytest.approx(
        action_type_weight["shared_total_delta_cap"]
        * reliability
        / action_type_weight["shared_active_route_count"]
    )
    assert action_type_weight["is_final_policy_contribution"] is False
    assert one_influence["baseline"]["final_logits"] == pytest.approx(
        one_influence["counterfactual"]["final_logits"]
    )

    zero = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        head_scales={"action_type": 0.0},
    )
    zero_influence = zero["decision_influence"]
    runtime_loo = zero["fusion"]["leave_one_out"]["action_type"]["runtime_path"]
    assert zero_influence["counterfactual"]["final_logits"] == pytest.approx(
        runtime_loo["policy_without_head_logits"]
    )
    assert zero_influence["parity"]["zero_x_matches_exact_leave_one_out"] == {
        "head": "action_type",
        "matches": True,
    }

    doubled = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        head_scales={"action_type": 2.0},
    )["decision_influence"]
    assert doubled["effective_scales"]["action_type"] == 2.0
    assert doubled["counterfactual"]["final_logits"] != pytest.approx(
        doubled["baseline"]["final_logits"]
    )
    assert torch.equal(
        model.decision_fusion.dedicated_routes["action_type"].network[-1].weight,
        parameter_before,
    )


def test_decision_influence_uses_exact_runtime_required_heads_fallback() -> None:
    class LegacyRuntimeFusion:
        required_heads = ("action_type",)
        dedicated_routes_enabled = True

        def __call__(
            self,
            _option_hidden: torch.Tensor,
            base_logits: torch.Tensor,
            *,
            state_sources: dict[str, torch.Tensor],
            option_sources: dict[str, torch.Tensor],
            dedicated_routes_active: bool,
        ) -> torch.Tensor:
            assert state_sources == {}
            assert dedicated_routes_active is True
            source = option_sources["action_type"]
            return base_logits + torch.tanh(source)

    fusion = LegacyRuntimeFusion()
    model = types.SimpleNamespace(
        decision_fusion=fusion,
        decision_fusion_runtime_enabled=True,
        decision_fusion_dedicated_routes_runtime_enabled=True,
    )
    option_hidden = torch.zeros((1, 2, 3))
    state = torch.zeros((1, 3))
    base_logits = torch.tensor([[0.1, 0.2]])
    option_sources = {"action_type": torch.tensor([[0.3, -0.4]])}
    baseline = fusion(
        option_hidden,
        base_logits,
        state_sources={},
        option_sources=option_sources,
        dedicated_routes_active=True,
    )
    without_head = fusion(
        option_hidden,
        base_logits,
        state_sources={},
        option_sources={"action_type": torch.zeros_like(option_sources["action_type"])},
        dedicated_routes_active=True,
    )
    fusion_payload = {
        "leave_one_out": {
            "action_type": {
                "runtime_path": {
                    "policy_without_head_logits": inference._tensor_to_json(
                        inference._option_tensor(without_head, 2)
                    )
                }
            }
        }
    }

    def inspect(scale: float) -> dict[str, object]:
        return inference._decision_influence_payload(
            model=model,
            option_hidden=option_hidden,
            state=state,
            base_logits=base_logits,
            baseline_final_logits=baseline,
            state_sources={},
            option_sources=option_sources,
            requested_scales={"action_type": scale},
            target=0,
            option_count=2,
            fusion_payload=fusion_payload,
        )

    one = inspect(1.0)
    zero = inspect(0.0)
    doubled = inspect(2.0)
    assert one["availability"]["available"] is True
    assert one["eligible_heads"] == ["action_type"]
    assert one["head_inventory_source"] == "required_heads"
    assert one["parity"]["one_x_matches_runtime_baseline"] is True
    assert zero["parity"]["zero_x_matches_exact_leave_one_out"] == {
        "head": "action_type",
        "matches": True,
    }
    assert doubled["counterfactual"]["final_logits"] != pytest.approx(
        doubled["baseline"]["final_logits"]
    )


def test_decision_influence_unknown_head_is_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    payload = inference.inspect_replay_step(
        model=_h10_model(monkeypatch),
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        head_scales={"not_a_head": 1.5},
    )["decision_influence"]

    assert payload["availability"]["available"] is False
    assert payload["invalid_heads"] == ["not_a_head"]


def test_legacy_heads_and_fusion_are_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _legacy_model(monkeypatch)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
    )

    assert payload["availability"]["available"] is True
    assert payload["heads"]["action_q"]["availability"]["available"] is False
    assert payload["heads"]["action_q"]["raw_values"] is None
    assert payload["heads"]["combo_state"]["availability"]["available"] is False
    assert payload["fusion"]["availability"]["available"] is False
    influence = payload["heads"]["value"]["policy_influence"]
    assert influence["availability"]["available"] is False
    assert "decision-fusion" in influence["availability"]["reason"]


def test_matchup_adapter_active_trace_has_exact_no_adapter_policy_effect(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, matchup_adapters_enabled=True)
    _materialize_trained_adapter(model)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        router=_Router(),
        submitted_runtime_activation=_submitted_runtime_activation(),
    )

    adapter = payload["adapter"]
    assert adapter["runtime_enabled"] is True
    assert adapter["decision_route_active"] is True
    assert adapter["runtime_activation"]["status"] == "applied"
    assert adapter["runtime_activation"]["evaluation_basis"] == (
        "checksum_bound_causal_re_evaluation"
    )
    assert adapter["runtime_activation"]["historical_activation_recorded"] is False
    assert (
        adapter["runtime_activation"]["cached_model_state_restored_after_request"]
        is True
    )
    assert adapter["route"] == 1
    assert adapter["slot"] == 1
    assert adapter["matched_archetype"]
    assert adapter["route_reliability"]["availability"]["available"] is True
    influence = adapter["policy_influence"]
    assert influence["availability"]["available"] is True
    assert influence["method"] == (
        "full_final_policy_minus_policy_without_matchup_adapter_route"
    )
    assert influence["sign_convention"] == (
        "full_final_policy_minus_policy_without_matchup_adapter_route"
    )
    loo = influence["leave_one_adapter_out"]
    assert sum(loo["effect_probabilities"]) == pytest.approx(0.0, abs=1e-6)
    assert influence["selected_option_probability_delta"] == pytest.approx(
        loo["effect_probabilities"][influence["selected_option_index"]]
    )
    assert model.matchup_adapter_bank.enabled is False


def test_submitted_runtime_unknown_route_is_exact_bypass_not_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, matchup_adapters_enabled=True)
    _materialize_trained_adapter(model)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        router=_UnknownRouter(),
        submitted_runtime_activation=_submitted_runtime_activation(),
    )

    adapter = payload["adapter"]
    assert adapter["runtime_enabled"] is True
    assert adapter["runtime_activation"]["applied"] is True
    assert adapter["route"] == -1
    assert adapter["decision_route_active"] is False
    reason = adapter["policy_influence"]["availability"]["reason"]
    assert "unknown/bypass route" in reason
    assert "runtime-disabled" not in reason
    assert model.matchup_adapter_bank.enabled is False


def test_no_verified_tree_forces_adapter_disabled_and_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, matchup_adapters_enabled=True)
    _materialize_trained_adapter(model)
    # Prove the serialized flag is not accepted as serving authority.
    model.matchup_adapter_bank.enabled = True

    payload = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        router=None,
        submitted_runtime_activation=None,
    )

    adapter = payload["adapter"]
    assert adapter["runtime_enabled"] is False
    assert adapter["decision_route_active"] is False
    activation = adapter["runtime_activation"]
    assert activation["status"] == "unavailable"
    assert activation["applied"] is False
    assert "startup activation evidence" in activation["reason"]
    assert activation["cached_model_state_restored_after_request"] is True
    assert model.matchup_adapter_bank.enabled is True


def test_submitted_runtime_activation_restores_cached_model_after_failure(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, matchup_adapters_enabled=True)
    _materialize_trained_adapter(model)
    previous_requires_grad = tuple(
        parameter.requires_grad for parameter in model.matchup_adapter_bank.parameters()
    )

    def fail_decode(*_args: object, **_kwargs: object) -> torch.Tensor:
        assert model.matchup_adapter_bank.enabled is True
        raise RuntimeError("fixture decode failure")

    monkeypatch.setattr(model, "decode_options", fail_decode)
    with pytest.raises(inference.ReplayInspectionError, match="model inference failed"):
        inference.inspect_replay_step(
            model=model,
            replay=_replay(),
            acting_seat=0,
            env_step=0,
            own_deck=[1] * 60,
            router=_Router(),
            submitted_runtime_activation=_submitted_runtime_activation(),
        )

    assert model.matchup_adapter_bank.enabled is False
    assert (
        tuple(
            parameter.requires_grad
            for parameter in model.matchup_adapter_bank.parameters()
        )
        == previous_requires_grad
    )


def test_router_reconstruction_observes_every_prior_decision_causally(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, matchup_adapters_enabled=True)
    _materialize_trained_adapter(model)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_two_decision_replay(),
        acting_seat=0,
        env_step=2,
        own_deck=[1] * 60,
        router=_CausalRouter(),
        submitted_runtime_activation=_submitted_runtime_activation(),
    )

    assert payload["replay"]["history_length"] == 2
    assert payload["replay"]["router"]["route"] == 1
    assert payload["replay"]["router"]["audit"]["observations"] == 2
    assert payload["adapter"]["decision_route_active"] is True


def test_prior_is_first_prompt_never_enters_router_or_model_history(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, matchup_adapters_enabled=True)
    _materialize_trained_adapter(model)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_is_first_then_decision_replay(),
        acting_seat=0,
        env_step=2,
        own_deck=[1] * 60,
        router=_CausalRouter(),
        submitted_runtime_activation=_submitted_runtime_activation(),
    )

    assert payload["replay"]["history_length"] == 1
    assert payload["replay"]["router"]["route"] == -1
    assert payload["replay"]["router"]["audit"]["observations"] == 1
    assert payload["adapter"]["runtime_enabled"] is True
    assert payload["adapter"]["decision_route_active"] is False


def test_is_first_prompt_can_be_scored_as_explicit_hypothetical_model_forward(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch)

    unavailable = inference.inspect_replay_step(
        model=model,
        replay=_is_first_then_decision_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
    )
    assert unavailable["availability"]["available"] is False
    assert "before the neural model runs" in unavailable["availability"]["reason"]

    hypothetical = inference.inspect_replay_step(
        model=model,
        replay=_is_first_then_decision_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
        router=_Router(),
        allow_setup_prompt_model_forward=True,
    )

    assert hypothetical["availability"]["available"] is True
    assert hypothetical["provenance"]["reproduction_status"] == (
        "hypothetical_model_forward_not_submitted_runtime"
    )
    assert hypothetical["replay"]["hypothetical_setup_prompt"] is True
    assert hypothetical["replay"]["history_length"] == 1
    assert hypothetical["replay"]["router"]["route"] is None
    assert hypothetical["replay"]["router"]["availability"]["available"] is False
    assert len(hypothetical["policy"]["probabilities"]) == 2
    assert sum(hypothetical["policy"]["probabilities"]) == pytest.approx(1.0)


def test_leave_one_out_uses_the_final_latent_policy_path(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _h10_model(monkeypatch, latent_action_authority=True)
    with torch.no_grad():
        assert model.latent_lookahead is not None
        # Make the normally zero-safe policy-aid branch observably active.
        model.latent_lookahead.policy_aid.bias.fill_(1.0)

    payload = inference.inspect_replay_step(
        model=model,
        replay=_replay(),
        acting_seat=0,
        env_step=0,
        own_deck=[1] * 60,
    )

    fusion = payload["fusion"]
    assert payload["policy"]["final_logits"] == pytest.approx(
        fusion["actual_final_logits"]
    )
    assert payload["policy"]["fusion_logits_before_latent"] != pytest.approx(
        fusion["actual_final_logits"]
    )
    loo = fusion["leave_one_out"]["value"]
    assert "runtime_path" in loo
    assert "runtime_path_before_latent" in loo


def test_parameter_inventory_summary_histogram_and_bounded_slice() -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([[1.0, 0.0], [-2.0, 3.0]]))
    )
    model.register_buffer("counter", torch.tensor([4, 5], dtype=torch.int64))
    inspector = ParameterInspector(model)

    inventory = inspector.inventory()
    assert inventory["parameter_count"] == 1
    assert inventory["buffer_count"] == 1
    assert {item["name"] for item in inventory["tensors"]} == {"weight", "counter"}

    summary = inspector.summary("weight")
    assert summary["shape"] == [2, 2]
    assert summary["finite_count"] == 4
    assert summary["minimum"] == -2.0
    assert summary["maximum"] == 3.0
    assert summary["zero_fraction"] == pytest.approx(0.25)

    histogram = inspector.histogram("weight", bins=4)
    assert histogram["availability"]["available"] is True
    assert sum(histogram["counts"]) == 4
    bounded = inspector.slice("weight", offset=1, limit=2)
    assert bounded["values"] == [0.0, -2.0]
    assert bounded["next_offset"] == 3
    with pytest.raises(ParameterInspectionError, match="unknown model tensor"):
        inspector.summary("not-a-tensor")
    with pytest.raises(ParameterInspectionError, match="slice limit"):
        inspector.slice("weight", limit=4097)


def test_guide_shadow_ranks_without_policy_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide = types.SimpleNamespace(
        GUIDE_VERSION="plain-english-fixture/v1",
        guide_scores=lambda _obs, _candidates, *, deck, force_enabled: (
            [0.25, 1.5, -0.5] if deck == [7] * 60 and force_enabled else None
        ),
    )
    monkeypatch.setattr(inference.deck_guides, "_GUIDES", {"fixture": guide})

    payload = inference._guide_shadow_payload(
        observation={"current": {}},
        candidates=[[0], [1], [2]],
        deck=[7] * 60,
        recorded_index=0,
        model_index=1,
    )

    assert payload["availability"]["available"] is True
    assert payload["guide_id"] == "fixture"
    assert payload["recommended_index"] == 1
    assert payload["agrees_with_recorded_action"] is False
    assert payload["agrees_with_model_action"] is True
    assert payload["policy_authority"] is False
    assert payload["policy_logit_delta"] == 0.0
    assert payload["historical_execution"] is False


def test_exact_submitted_guide_policy_becomes_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("poke_bot.submission_guide_policy")
    module.load_config = lambda: {
        "schema": "poke_bot.submission_guide_decision_policy/v1",
        "guide_logit_weight": 0.1,
    }

    def select_index(**kwargs: object) -> tuple[int, dict[str, object]]:
        probabilities = kwargs["model_policy"]
        assert isinstance(probabilities, list)
        adjusted = [
            math.log(max(float(probabilities[0]), 1e-12)),
            math.log(max(float(probabilities[1]), 1e-12)) + 0.1,
        ]
        return 1, {
            "schema": "poke_bot.submission_guide_decision_policy/v1",
            "guide_available": True,
            "guide_selected": True,
            "guide_logit_weight": 0.1,
            "normalized_guide_scores": [0.0, 1.0],
            "adjusted_log_scores": adjusted,
            "selected_index": 1,
            "reason": "bounded_guide_logit_bonus",
        }

    module.select_index = select_index
    monkeypatch.setitem(sys.modules, "poke_bot.submission_guide_policy", module)
    logits = torch.tensor([0.05, 0.0])
    neural_probabilities = torch.softmax(logits, dim=-1)

    payload, submitted_logits, probabilities, selected, bonus = (
        inference._submitted_runtime_policy_payload(
            observation={"current": {}},
            candidates=[[0], [1]],
            deck=[7] * 60,
            neural_logits=logits,
            neural_probabilities=neural_probabilities,
            neural_index=0,
        )
    )

    assert payload["availability"]["available"] is True
    assert payload["policy_authority"] is True
    assert payload["applied"] is True
    assert payload["changed_neural_choice"] is True
    assert selected == 1
    assert bonus.tolist() == pytest.approx([0.0, 0.1])
    assert int(torch.argmax(submitted_logits).item()) == 1
    assert probabilities.tolist() == pytest.approx(
        torch.softmax(submitted_logits, dim=-1).tolist()
    )


def test_cpu_cache_requires_digest_and_reuses_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"fixture")
    expected = "sha256:" + "a" * 64
    calls: list[Path] = []

    def fake_loader(loaded_path: Path, *, device: torch.device) -> torch.nn.Module:
        assert device.type == "cpu"
        calls.append(Path(loaded_path))
        return torch.nn.Linear(2, 1)

    fake_train = types.ModuleType("poke_bot.train")
    fake_train.load_model_from_checkpoint = fake_loader
    monkeypatch.setitem(sys.modules, "poke_bot.train", fake_train)
    monkeypatch.setattr(
        inference.checkpoint, "checkpoint_digest", lambda _path: expected
    )

    cache = inference.VerifiedCpuModelCache()
    first = cache.load(path, expected)
    second = cache.load(path, expected)
    assert first is second
    assert len(calls) == 1
    assert first.model.training is False
    with pytest.raises(
        inference.ReplayInspectionError, match="sha256 checkpoint digest"
    ):
        cache.load(path, "")
