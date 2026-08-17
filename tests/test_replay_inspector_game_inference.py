from __future__ import annotations

import gc
import math
import weakref
from typing import Any

import pytest
import torch

from poke_bot import config, features
from poke_bot.model import build_model
from replay_inspector import inference


def _stub_feature_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
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


def _two_decision_replay() -> dict[str, object]:
    first = {
        "current": {"turn": 3, "yourIndex": 0},
        "select": {"context": 1, "option": [{}, {}], "minCount": 1, "maxCount": 1},
    }
    second = {
        "current": {"turn": 4, "yourIndex": 0},
        "select": {"context": 1, "option": [{}, {}], "minCount": 1, "maxCount": 1},
    }
    return {
        "steps": [
            [{"status": "ACTIVE", "observation": first}, {}],
            [{"action": [1, 0]}, {}],
            [{"status": "ACTIVE", "observation": second}, {}],
            [{"action": [1]}, {}],
        ]
    }


def _is_first_then_decision_replay() -> dict[str, object]:
    setup = {
        "current": {"turn": 0, "yourIndex": 0},
        "select": {
            "context": "IsFirst",
            "option": [{}, {}],
            "minCount": 1,
            "maxCount": 1,
        },
    }
    decision = {
        "current": {"turn": 1, "yourIndex": 0},
        "select": {"context": 1, "option": [{}, {}], "minCount": 1, "maxCount": 1},
    }
    return {
        "steps": [
            [{"status": "ACTIVE", "observation": setup}, {}],
            [{"action": [1]}, {}],
            [{"status": "ACTIVE", "observation": decision}, {}],
            [{"action": [1]}, {}],
        ]
    }


def _many_decision_replay(count: int) -> dict[str, object]:
    steps: list[list[dict[str, object]]] = []
    for index in range(count):
        observation = {
            "current": {"turn": index + 5, "yourIndex": 0},
            "select": {
                "context": 1,
                "option": [{}, {}, {}],
                "minCount": 1,
                "maxCount": 1,
            },
        }
        steps.append([{"status": "ACTIVE", "observation": observation}, {}])
        steps.append([{"action": [1]}, {}])
    return {"steps": steps}


def _submitted_runtime_activation() -> dict[str, object]:
    return {
        "basis": "checksum_bound_submitted_startup",
        "runtime_parity_verified": True,
        "runtime_identity_verified": True,
        "runtime_source_tree_sha256": "sha256:" + "a" * 64,
        "matchup_tree_verified": True,
        "matchup_tree_sha256": "sha256:" + "b" * 64,
        "submitted_startup_behavior_verified": True,
        "submitted_startup_behavior": "packaged_matchup_tree_enables_policy_agent_runtime",
    }


class _CausalRouter:
    def __init__(self) -> None:
        self.observed: list[object] = []

    @property
    def candidate_model_route(self) -> int:
        return 1 if len(self.observed) >= 2 else -1

    def fork(self) -> _CausalRouter:
        return _CausalRouter()

    def observe(self, observation: object, **_kwargs: object) -> None:
        self.observed.append(observation)

    def snapshot(self, **_kwargs: object) -> dict[str, object]:
        return {"observations": len(self.observed), "route": self.candidate_model_route}


@pytest.fixture
def feature_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference.features, "assert_info_set", lambda _obs: None)
    monkeypatch.setattr(
        inference.features, "build_board_tokens", lambda _obs, _deck: _board_tokens()
    )
    monkeypatch.setattr(inference.features, "build_option_tokens", _option_tokens)

    def factorized(obs: dict[str, object], _action: list[int]):
        turn = (obs.get("current") or {}).get("turn")
        if turn == 3:
            return [([[0], [1]], 1), ([[0], [2]], 0)]
        return [([[0], [1], [2]], 1)]

    monkeypatch.setattr(
        inference.features, "factorized_teacher_forcing_stages", factorized
    )


def _assert_equivalent(expected: Any, actual: Any, *, path: str = "payload") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_equivalent(expected_value, actual[key], path=f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _assert_equivalent(expected_value, actual_value, path=f"{path}[{index}]")
    elif isinstance(expected, float):
        assert isinstance(actual, float)
        assert math.isfinite(expected) == math.isfinite(actual)
        if math.isfinite(expected):
            assert actual == pytest.approx(expected, abs=1e-5, rel=1e-5)
    else:
        assert actual == expected, path


def test_game_reuses_history_for_factorized_stages_and_matches_step_traces(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _model(monkeypatch)
    replay = _two_decision_replay()
    addresses = [(0, 0), (0, 1), (2, 0)]
    history_calls: list[int] = []
    original_history = model.forward_history_batch

    def counted_history(*args: object, **kwargs: object):
        histories = args[0]
        history_calls.append(len(histories))
        return original_history(*args, **kwargs)

    monkeypatch.setattr(model, "forward_history_batch", counted_history)

    game = inference.inspect_replay_game(
        model=model,
        replay=replay,
        acting_seat=0,
        addresses=addresses,
        own_deck=[1] * 60,
        router_factory=_CausalRouter,
        checkpoint_digest="sha256:test-model",
        batch_size=8,
    )

    assert list(game) == addresses
    assert history_calls == [2]
    assert game[(0, 0)]["replay"]["router"]["route"] == -1
    assert game[(0, 1)]["replay"]["router"] == game[(0, 0)]["replay"]["router"]
    assert game[(2, 0)]["replay"]["router"]["route"] == 1
    assert len(game[(2, 0)]["replay"]["legal_candidates"]) == 3

    for address in addresses:
        individual = inference.inspect_replay_step(
            model=model,
            replay=replay,
            acting_seat=0,
            env_step=address[0],
            factorized_stage=address[1],
            own_deck=[1] * 60,
            router_factory=_CausalRouter,
            checkpoint_digest="sha256:test-model",
        )
        _assert_equivalent(individual, game[address])


def test_game_all_addresses_uses_timeline_and_chunks_history_by_physical_step(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _model(monkeypatch)
    replay = _two_decision_replay()
    history_calls: list[int] = []
    original_history = model.forward_history_batch

    def counted_history(*args: object, **kwargs: object):
        histories = args[0]
        history_calls.append(len(histories))
        return original_history(*args, **kwargs)

    monkeypatch.setattr(model, "forward_history_batch", counted_history)

    game = inference.inspect_replay_game(
        model=model,
        replay=replay,
        acting_seat=0,
        own_deck=[1] * 60,
        router_factory=_CausalRouter,
        batch_size=1,
    )

    assert list(game) == [(0, 0), (0, 1), (2, 0)]
    assert history_calls == [1, 1]
    for address, payload in game.items():
        assert payload["availability"]["available"] is True
        assert payload["replay"]["env_step"] == address[0]
        assert payload["replay"]["factorized_stage"] == address[1]


def test_game_keeps_hypothetical_setup_on_forced_bypass_runtime(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    model = _model(monkeypatch)

    game = inference.inspect_replay_game(
        model=model,
        replay=_is_first_then_decision_replay(),
        acting_seat=0,
        own_deck=[1] * 60,
        router_factory=_CausalRouter,
        submitted_runtime_activation=_submitted_runtime_activation(),
        allow_setup_prompt_model_forward=True,
    )

    assert list(game) == [(0, 0), (2, 0)]
    setup = game[(0, 0)]
    ordinary = game[(2, 0)]
    assert setup["replay"]["hypothetical_setup_prompt"] is True
    assert setup["adapter"]["runtime_activation"]["requested"] is False
    assert ordinary["replay"]["hypothetical_setup_prompt"] is False
    assert ordinary["adapter"]["runtime_activation"]["requested"] is True


def test_game_releases_tensor_chunks_before_next_history_batch(
    monkeypatch: pytest.MonkeyPatch, feature_stubs: None
) -> None:
    """Whole-game output is JSON; model tensors must not grow with game size."""

    model = _model(monkeypatch)
    original_history = model.forward_history_batch
    original_decode = model.decode_options
    history_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    hidden_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    history_alive_before: list[int] = []
    hidden_alive_before: list[int] = []
    history_batch_sizes: list[int] = []
    option_batch_sizes: list[int] = []

    def tracked_history(*args: object, **kwargs: object):
        gc.collect()
        history_alive_before.append(sum(ref() is not None for ref in history_refs))
        histories = args[0]
        history_batch_sizes.append(len(histories))
        result = original_history(*args, **kwargs)
        state = result["state_vec"]
        assert isinstance(state, torch.Tensor)
        history_refs.append(weakref.ref(state))
        return result

    def tracked_decode(*args: object, **kwargs: object):
        return_hidden = kwargs.get("return_hidden") is True
        if return_hidden:
            gc.collect()
            hidden_alive_before.append(sum(ref() is not None for ref in hidden_refs))
            option_batch_sizes.append(len(args[0]))
        result = original_decode(*args, **kwargs)
        if return_hidden:
            assert isinstance(result, tuple)
            hidden_refs.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(model, "forward_history_batch", tracked_history)
    monkeypatch.setattr(model, "decode_options", tracked_decode)

    game = inference.inspect_replay_game(
        model=model,
        replay=_many_decision_replay(6),
        acting_seat=0,
        own_deck=[1] * 60,
        batch_size=2,
    )

    assert len(game) == 6
    assert history_batch_sizes == [2, 2, 2]
    assert option_batch_sizes == [2, 2, 2]
    assert history_alive_before == [0, 0, 0]
    assert hidden_alive_before == [0, 0, 0]
    gc.collect()
    assert all(ref() is None for ref in history_refs)
    assert all(ref() is None for ref in hidden_refs)
