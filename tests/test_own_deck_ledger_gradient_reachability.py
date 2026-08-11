"""Offline-only gradient reachability for the dormant own-deck successor.

This is deliberately an architecture/optimization test, not a serving switch:
every runtime flag below remains false.  ``model.train()`` is the explicit
offline path that lets the zero-safe residuals receive their first update.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
import torch.nn.functional as F

from poke_bot import config, features
from poke_bot.features import SparseVector
from poke_bot.model import build_model
from poke_bot.own_deck_ledger import OPTION_FEATURE_DIM, OwnDeckLedger


@pytest.fixture(autouse=True)
def _stub_feature_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this isolated model test independent of a cg installation."""

    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "encoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)


def _cfg(**overrides: object) -> config.ModelConfig:
    values: dict[str, object] = {
        "d_model": 16,
        "spatial_layers": 1,
        "temporal_layers": 1,
        "option_decoder_layers": 1,
        "n_heads": 4,
        "ff_dim": 32,
        "max_context": 8,
        "temporal_pos": "rope",
        "decision_context": "history",
        "kv_cache": False,
        "dropout": 0.0,
        # Physical successor architecture is present, but every serving gate
        # remains off.  Training mode below is the only activation authority.
        "own_deck_ledger_enabled": True,
        "own_deck_ledger_runtime_enabled": False,
        "expanded_heads_enabled": True,
        "decision_fusion_enabled": True,
        "decision_fusion_runtime_enabled": False,
        "decision_fusion_dedicated_routes_enabled": True,
        "decision_fusion_dedicated_routes_runtime_enabled": False,
        "visible_tutor_completion_head_enabled": True,
        "terminal_conversion_head_enabled": True,
        "visible_tutor_completion_route_enabled": True,
        "visible_tutor_completion_route_runtime_enabled": False,
        "terminal_conversion_route_enabled": True,
        "terminal_conversion_route_runtime_enabled": False,
    }
    values.update(overrides)
    return config.ModelConfig(**values)


def _model():
    return build_model(
        _cfg(),
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )


def _board() -> SparseVector:
    result = SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        result.word_start()
    return result


def _options(n: int = 3) -> SparseVector:
    result = SparseVector()
    for index in range(n):
        result.word_start()
        result.add((index + 1) % 32, 1.0)
    return result


def _valid_snapshot_and_menu():
    """Build an actual causal snapshot plus a visible tutor/looking menu."""

    ledger = OwnDeckLedger([1] * 30 + [2] * 30)
    observation = {
        "current": {
            "yourIndex": 0,
            "looking": [{"id": 2, "serial": 202}],
            "players": [
                {
                    "hand": [{"id": 1, "serial": 101}],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    # The exposed looking card is a physical transit-zone
                    # card, so it subtracts once from the live draw pile.
                    "deckCount": 52,
                },
                {
                    "hand": [],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [],
                },
            ],
        },
        "select": {
            "deck": [{"id": 2, "serial": 201}],
            "option": [
                {"type": 10, "area": 1, "index": 0, "playerIndex": 0},
                {"type": 10, "area": 12, "index": 0, "playerIndex": 0},
                {"type": 0},
            ],
        },
    }
    snapshot = ledger.observe(observation)
    assert snapshot.integrity_ok is True
    assert snapshot.fail_closed is False
    menu = snapshot.option_features(observation, [[0], [1], []])
    assert len(menu) == 3
    assert all(len(row) == OPTION_FEATURE_DIM for row in menu)
    assert menu[0] != (0.0,) * OPTION_FEATURE_DIM
    assert menu[1] != (0.0,) * OPTION_FEATURE_DIM
    assert menu[2] == (0.0,) * OPTION_FEATURE_DIM
    return snapshot, menu


def _loss(outputs: dict[str, object]) -> torch.Tensor:
    """Give every public head a small, deterministic offline loss signal."""

    policy = outputs["policy_logits"]
    assert isinstance(policy, torch.Tensor)
    loss = F.cross_entropy(policy, torch.tensor([1], dtype=torch.long))
    for name in (
        "value",
        "aux_logits",
        "opp_hand_logits",
        "opp_remainder_logits",
        "lethal_threat_logits",
        "prize_race_pred",
        "visible_tutor_completion_logits",
        "terminal_conversion_logits",
    ):
        value = outputs[name]
        assert isinstance(value, torch.Tensor)
        # ``mean`` has a nonzero output derivative even if a zero-safe route
        # happens to produce all zeros on the initial update.
        loss = loss + 0.01 * value.float().mean()
    return loss


def _nonzero_gradient(parameter: torch.nn.Parameter | None, label: str) -> None:
    assert parameter is not None, f"missing parameter: {label}"
    assert parameter.grad is not None, f"no gradient: {label}"
    assert torch.count_nonzero(parameter.grad).item() > 0, f"zero gradient: {label}"


def _any_nonzero_gradient(module: torch.nn.Module, label: str) -> None:
    parameters: Iterable[torch.nn.Parameter] = module.parameters()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    ), f"no nonzero gradient in {label}"


def _gradient_reaches(
    signal: torch.Tensor,
    parameter: torch.nn.Parameter,
    label: str,
) -> None:
    gradient = torch.autograd.grad(
        signal.float().mean(),
        parameter,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert gradient is not None, f"disconnected path: {label}"
    assert torch.count_nonzero(gradient).item() > 0, f"zero path gradient: {label}"


def test_offline_ledger_gradients_reach_all_successor_paths_without_serving_activation() -> None:
    """One warm-up step opens zero-safe paths; the second proves reachability."""

    torch.manual_seed(101)
    model = _model()
    snapshot, menu = _valid_snapshot_and_menu()
    histories = [[_board(), _board()]]
    options = [_options()]
    ledger_histories = [[snapshot, snapshot]]

    # A physically present successor must remain an exact serving no-op while
    # all receipt/runtime switches are false.
    model.eval()
    with torch.no_grad():
        baseline = model.forward_history_batch(histories, options, n_options=[3])
        dormant = model.forward_history_batch(
            histories,
            options,
            n_options=[3],
            ledger_histories=ledger_histories,
            ledger_option_features=[menu],
        )
    for name in (
        "policy_logits",
        "value",
        "aux_logits",
        "opp_hand_logits",
        "opp_remainder_logits",
        "lethal_threat_logits",
        "prize_race_pred",
        "visible_tutor_completion_logits",
        "terminal_conversion_logits",
    ):
        torch.testing.assert_close(dormant[name], baseline[name], rtol=0.0, atol=0.0)
    assert model.own_deck_ledger_runtime_enabled is False
    assert model.decision_fusion_runtime_enabled is False
    assert model.decision_fusion_dedicated_routes_runtime_enabled is False
    assert model.visible_tutor_completion_route_runtime_enabled is False
    assert model.terminal_conversion_route_runtime_enabled is False

    adapter = model.own_deck_ledger_adapter
    option_adapter = model.own_deck_ledger_option_adapter
    fusion = model.decision_fusion
    tutor_head = model.visible_tutor_completion_head
    terminal_head = model.terminal_conversion_head
    tutor_route = model.visible_tutor_completion_route
    terminal_route = model.terminal_conversion_route
    assert adapter is not None
    assert option_adapter is not None
    assert fusion is not None
    assert tutor_head is not None
    assert terminal_head is not None
    assert tutor_route is not None
    assert terminal_route is not None

    # The final projections are intentionally exact-zero on construction.  A
    # first ordinary offline optimizer step trains just those exits; requiring
    # lower-layer gradients before that would incorrectly reject zero-safe
    # migration behavior.
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    warm = model.forward_history_batch(
        histories,
        options,
        n_options=[3],
        ledger_histories=ledger_histories,
        ledger_option_features=[menu],
    )
    optimizer.zero_grad(set_to_none=True)
    _loss(warm).backward()
    _nonzero_gradient(adapter.output.weight, "shared ledger final projection")
    _nonzero_gradient(option_adapter.network[-1].weight, "option ledger final projection")
    _nonzero_gradient(fusion.residual[-1].weight, "fusion final projection")
    _nonzero_gradient(tutor_route.network[-1].weight, "tutor route final projection")
    _nonzero_gradient(terminal_route.network[-1].weight, "terminal route final projection")
    for name, route in fusion.dedicated_routes.items():
        _nonzero_gradient(route.network[-1].weight, f"fusion route final projection: {name}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # On the second pass, prove direct differentiated paths from the shared
    # ledger adapter into policy/value/legacy belief heads and the expanded
    # state heads.  This does not rely on a nonzero serving flag.
    output = model.forward_history_batch(
        histories,
        options,
        n_options=[3],
        ledger_histories=ledger_histories,
        ledger_option_features=[menu],
    )
    shared_lower = adapter.trunk[1].weight
    for name in (
        "policy_logits",
        "value",
        "aux_logits",
        "opp_hand_logits",
        "opp_remainder_logits",
        "lethal_threat_logits",
        "prize_race_pred",
        "visible_tutor_completion_logits",
        "terminal_conversion_logits",
    ):
        value = output[name]
        assert isinstance(value, torch.Tensor)
        _gradient_reaches(value, shared_lower, f"shared ledger -> {name}")

    state_vec = output["state_vec"]
    spatial_memory = output["spatial_memory"]
    assert isinstance(state_vec, torch.Tensor)
    assert isinstance(spatial_memory, torch.Tensor)
    decoded = model.decode_options(
        options,
        spatial_memory,
        state_vec,
        n_options=[3],
        decision_fusion_state_vec=state_vec,
        ledger_option_features=[menu],
        return_hidden=True,
    )
    assert isinstance(decoded, tuple)
    _decoded_logits, option_hidden = decoded
    expanded_state = model.expanded_state_logits(state_vec)
    expanded_option = model.expanded_option_logits(option_hidden)
    for name, value in expanded_state.items():
        _gradient_reaches(value, shared_lower, f"shared ledger -> expanded state {name}")
    option_lower = option_adapter.network[1].weight
    _gradient_reaches(option_hidden, option_lower, "option ledger -> decoder hidden")
    for name, value in expanded_option.items():
        _gradient_reaches(value, option_lower, f"option ledger -> expanded option {name}")

    # Include explicit expanded-head and route signals so the post-warmup
    # backward pass independently checks each training-only route, rather than
    # depending on a particular fusion residual's numerical scale.
    second_loss = _loss(output)
    for value in expanded_state.values():
        second_loss = second_loss + 0.002 * value.float().mean()
    for value in expanded_option.values():
        second_loss = second_loss + 0.002 * value.float().mean()
    for value in model.decision_fusion_route_deltas(option_hidden, state_vec).values():
        second_loss = second_loss + 0.002 * value.float().mean()
    for value in model.own_deck_option_route_deltas(option_hidden).values():
        second_loss = second_loss + 0.002 * value.float().mean()
    second_loss.backward()

    # Shared ledger input reaches the common temporal trunk and every legacy
    # policy/value/belief output.  The expanded state heads are consumed by
    # fusion during the policy path; the option heads share the decoder path.
    _nonzero_gradient(adapter.trunk[1].weight, "shared ledger lower trunk")
    _nonzero_gradient(adapter.card_embedding.weight, "shared ledger card identity")
    _nonzero_gradient(option_adapter.network[1].weight, "option ledger lower trunk")
    _any_nonzero_gradient(model.temporal_blocks, "temporal history encoder")
    _any_nonzero_gradient(model.option_decoder, "option decoder")
    for name in (
        "policy_head",
        "value_head",
        "aux_head",
        "opp_hand_head",
        "opp_remainder_head",
        "lethal_threat_head",
        "prize_race_head",
    ):
        _any_nonzero_gradient(getattr(model, name), name)
    for name in (
        "tactical_outcome_head",
        "opponent_response_head",
        "resource_forecast_head",
        "game_phase_head",
        "outcome_distribution_head",
        "remaining_turns_head",
        "action_q_head",
        "action_type_head",
        "action_target_head",
        "action_resource_head",
        "action_utility_head",
    ):
        _any_nonzero_gradient(getattr(model, name), name)
    _nonzero_gradient(fusion.residual[0].weight, "fusion lower residual")
    for name, projection in fusion.state_projections.items():
        _nonzero_gradient(projection.weight, f"fusion state projection: {name}")
    for name, route in fusion.dedicated_routes.items():
        _nonzero_gradient(route.network[0].weight, f"fusion route lower projection: {name}")
    _nonzero_gradient(tutor_head.projection.weight, "visible tutor head")
    _nonzero_gradient(terminal_head.projection.weight, "terminal conversion head")
    _nonzero_gradient(tutor_route.network[0].weight, "visible tutor training route")
    _nonzero_gradient(terminal_route.network[0].weight, "terminal conversion training route")

    # The offline test did not mutate serving authority as a side effect.
    assert model.own_deck_ledger_runtime_enabled is False
    assert model.decision_fusion_runtime_enabled is False
    assert model.decision_fusion_dedicated_routes_runtime_enabled is False
    assert model.visible_tutor_completion_route_runtime_enabled is False
    assert model.terminal_conversion_route_runtime_enabled is False
