"""Engine-fixture and metamorphic coverage for the isolated r298 rule targets.

These tests deliberately use only the public selected-action bridge.  The
fixture is *not* evidence of a pinned competition-engine execution: the
separate Elmo runner must supply that evidence, and the aggregate validator
rejects fixture-only claims.  Keeping this boundary explicit lets us exercise
prompt-chain semantics locally without silently promoting a mock into a game
rules authority.
"""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from poke_bot.alakazam_public_rule_adapter_r298 import (
    MAX_NUMBER_VALUE,
    MAX_OPTION_COUNT,
    PublicRuleAdapterError,
    build_public_rule_representation,
    public_rule_observation_fingerprint,
    sanitize_public_observation,
    semantic_option_key,
)
from poke_bot.alakazam_rule_aux_heads_r298 import (
    R298RuleAuxHeadsConfig,
    R298RuleAuxiliaryHeads,
)
from poke_bot.alakazam_simulator_rule_targets_r298 import (
    R298_CANONICAL_SIMULATOR,
    DeterministicPromptChain,
    PromptChainStep,
    SimulatorRuleTargetError,
    assert_public_target_invariance,
    compile_simulator_rule_targets,
    prize_yield_from_public_card,
    rule_head_target_vectors,
)


def _card(
    card_id: int,
    serial: int,
    *,
    hp: int = 100,
    max_hp: int = 100,
    ex: bool | None = None,
    mega_ex: bool | None = None,
    energy: list[dict[str, Any]] | None = None,
    pre_evolution: list[dict[str, Any]] | None = None,
    appear_this_turn: bool | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": max_hp,
        "energyCards": list(energy or []),
        "tools": [],
    }
    if ex is not None:
        row["ex"] = ex
    if mega_ex is not None:
        row["megaEx"] = mega_ex
    if pre_evolution is not None:
        row["preEvolution"] = list(pre_evolution)
    if appear_this_turn is not None:
        row["appearThisTurn"] = appear_this_turn
    return row


def _player(
    *,
    active: list[dict[str, Any]],
    bench: list[dict[str, Any]] | None = None,
    hand: list[dict[str, Any]] | None = None,
    deck_count: int = 30,
    prize_count: int = 6,
    bench_max: int = 5,
) -> dict[str, Any]:
    return {
        "active": list(active),
        "bench": list(bench or []),
        "hand": list(hand or []),
        "handCount": len(hand or []),
        "discard": [],
        "prize": [None] * prize_count,
        "prizeCount": prize_count,
        "deckCount": deck_count,
        "benchMax": bench_max,
    }


def _observation(
    options: list[dict[str, Any]],
    *,
    deck_count: int = 30,
    own_bench: list[dict[str, Any]] | None = None,
    own_bench_max: int = 5,
    opponent_hand: list[dict[str, Any]] | None = None,
    own_prizes: int = 3,
    opponent_prizes: int = 3,
) -> dict[str, Any]:
    active = _card(
        10,
        100,
        hp=70,
        max_hp=140,
        energy=[
            {"id": 20, "serial": 200, "typedEnergyUnits": {"Psychic": 1, "Fire": 1}},
            {"id": 21, "serial": 201, "energyType": "Psychic"},
        ],
        pre_evolution=[{"id": 9, "serial": 99}],
        appear_this_turn=True,
    )
    return {
        "current": {
            "yourIndex": 0,
            "turn": 7,
            "turnActionCount": 1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "oncePerTurnActionFlags": {"telepathy": True},
            "result": -1,
            "players": [
                _player(
                    active=[active],
                    bench=own_bench
                    if own_bench is not None
                    else [_card(77, 1001), _card(77, 1002)],
                    hand=[_card(30, 300)],
                    deck_count=deck_count,
                    prize_count=own_prizes,
                    bench_max=own_bench_max,
                ),
                _player(
                    active=[_card(11, 101)],
                    hand=opponent_hand if opponent_hand is not None else [_card(901, 9001)],
                    prize_count=opponent_prizes,
                ),
            ],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": "Main",
            "type": "Card",
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 2,
            "remainEnergyCost": 3,
            "effect": {"effectId": 701},
            "option": copy.deepcopy(options),
        },
    }


@dataclass
class _InjectedSelectedActionFixture:
    """Narrow fixture matching the production selected-action bridge only."""

    chain: DeterministicPromptChain
    canonical_simulator_identity: Mapping[str, Any] = field(
        default_factory=lambda: R298_CANONICAL_SIMULATOR
    )
    calls: int = 0

    def resolve_selected_action_prompt_chain(
        self,
        *,
        public_observation: Mapping[str, Any],
        selected_action: Sequence[int],
    ) -> DeterministicPromptChain:
        self.calls += 1
        # The compiler hands an already-sanitized public view to the bridge;
        # no private deck / hand source is available at this seam.
        assert "deckOrder" not in str(public_observation)
        assert tuple(selected_action) == self.chain.root_action
        return self.chain


def _fixture_chain(
    before: dict[str, Any],
    *,
    selected: int = 0,
    events: list[PromptChainStep] | None = None,
) -> DeterministicPromptChain:
    if events is None:
        after = copy.deepcopy(before)
        after["current"]["turnActionCount"] += 1
        events = [
            PromptChainStep(
                before=copy.deepcopy(before),
                after=after,
                event_kind="attack",
                action=(selected,),
                strategic_decision=True,
                facts={"damage": 20},
            )
        ]
    return DeterministicPromptChain(
        root_action=(selected,),
        events=tuple(events),
        simulator=R298_CANONICAL_SIMULATOR,
    )


def _compile_with_fixture(
    observation: dict[str, Any],
    chain: DeterministicPromptChain,
    *,
    selected: int = 0,
) -> dict[str, Any]:
    fixture = _InjectedSelectedActionFixture(chain)
    result = compile_simulator_rule_targets(
        {"observation": observation, "action": [selected]},
        simulator=fixture,
        strict=True,
    )
    assert fixture.calls == 1
    return result


def test_public_semantics_preserve_rule_distinctions_without_text_or_ordinals() -> None:
    """Exercise the distinctions that are easy to accidentally coalesce."""

    options = [
        {"type": "Number", "number": 4},
        {"type": "Number", "number": 5},
        {"type": "Skill", "cardId": 77, "serial": 1001},
        {"type": "Skill", "cardId": 77, "serial": 1002},
        {"type": "Attack", "attackId": 44, "simulatorDiscriminator": "first"},
        {"type": "Attack", "attackId": 44, "simulatorDiscriminator": "second"},
        {
            "type": "Attach",
            "playerIndex": 0,
            "area": "Hand",
            "index": 0,
            "inPlayArea": "Bench",
            "inPlayIndex": 0,
            "count": 2,
            "energyIndex": 0,
        },
    ]
    observation = _observation(options)
    representation = build_public_rule_representation(observation)
    semantic_keys = [option.semantic_key_sha256 for option in representation.options]
    assert len(set(semantic_keys)) == len(semantic_keys)
    assert representation.selection["remain_damage_counter"] == 2
    assert representation.selection["remain_energy_cost"] == 3
    assert representation.options[-1].semantic["option"]["count"] == 2
    active = representation.state["players"]["acting"]["active"][0]["card"]
    assert active["max_hp"] == 140
    assert active["pre_evolution_card_ids"] == [9]
    assert active["appear_this_turn"] is True
    assert active["typed_energy_units"] == [
        ["energy_type:fire", 1],
        ["energy_type:psychic", 2],
    ]
    assert sanitize_public_observation(observation)["current"]["oncePerTurnActionFlags"] == {
        "telepathy": True
    }

    # YES/NO are context-and-effect sensitive semantic options, not generic
    # booleans.  Separate stages with a different effect must not collapse.
    yes = _observation([{"type": "Yes"}])
    yes_other_effect = copy.deepcopy(yes)
    yes_other_effect["select"]["context"] = "CoinHead"
    yes_other_effect["select"]["effect"] = {"effectId": 702}
    no = _observation([{"type": "No"}])
    assert semantic_option_key(yes, yes["select"]["option"][0]) != semantic_option_key(
        yes_other_effect, yes_other_effect["select"]["option"][0]
    )
    assert semantic_option_key(yes, yes["select"]["option"][0]) != semantic_option_key(
        no, no["select"]["option"][0]
    )

    with pytest.raises(PublicRuleAdapterError, match="exceeds"):
        build_public_rule_representation(
            _observation([{"type": "Number", "number": MAX_NUMBER_VALUE + 1}])
        )

    # The representation must reject an oversized legal surface rather than
    # silently truncating it.  Truncation could remove the selected legal
    # action or make two options collide after padding.
    with pytest.raises(PublicRuleAdapterError, match="exceeds"):
        build_public_rule_representation(
            _observation(
                [
                    {"type": "Number", "number": index}
                    for index in range(MAX_OPTION_COUNT + 1)
                ]
            )
        )


def test_prize_classes_apply_mega_precedence_and_visible_reduction() -> None:
    assert prize_yield_from_public_card({"ex": False, "megaEx": False}) == 1
    assert prize_yield_from_public_card({"ex": True, "megaEx": False}) == 2
    assert prize_yield_from_public_card({"ex": True, "megaEx": True}) == 3
    assert prize_yield_from_public_card({"ex": True}, visible_modifier={"reduction": 1}) == 1
    assert prize_yield_from_public_card({"ex": True, "megaEx": True}, visible_modifier={"reduction": 1}) == 2
    assert prize_yield_from_public_card({"ex": True}, visible_modifier={"exact_yield": 0}) == 0


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("deck_zero", {"deck_count": 0}),
        ("full_bench", {"own_bench": [_card(80 + index, 1100 + index) for index in range(5)]}),
        ("opponent_hand_zero", {"opponent_hand": []}),
    ],
)
def test_simulator_omitted_attack_is_never_synthesized(
    label: str, kwargs: dict[str, Any]
) -> None:
    """A fixture legal list with no ATTACK remains no ATTACK in targets."""

    observation = _observation(
        [
            {"type": "Skill", "cardId": 77, "serial": 1001},
            {"type": "End"},
        ],
        **kwargs,
    )
    chain = _fixture_chain(observation, selected=0)
    result = _compile_with_fixture(observation, chain)
    readiness = result["attack_readiness"]
    assert readiness["values"][:2] == [0.0, 0.0], label
    assert readiness["mask"][:2] == [True, True], label
    semantics = result["legal_option_semantics"]
    assert semantics["legal_option_set_authority"] == "simulator_emitted_only_no_synthesis"
    assert all(
        row["semantic"]["option"]["option_type"] != "attack"
        for row in semantics["options"]
    )


def test_nullifying_zero_forced_order_and_simultaneous_ko_credit_selected_attack() -> None:
    """The selected attack owns its forced prompt chain, but no forced target."""

    observation = _observation([{"type": "Attack", "attackId": 44}, {"type": "Skill", "cardId": 77, "serial": 1001}])
    after_attack = copy.deepcopy(observation)
    after_attack["current"]["turnActionCount"] = 2
    after_nullifying = copy.deepcopy(after_attack)
    after_ko = copy.deepcopy(after_nullifying)
    after_ko["current"]["players"][0]["prizeCount"] = 0
    after_promotion = copy.deepcopy(after_ko)
    after_draw = copy.deepcopy(after_promotion)
    after_draw["current"]["players"][0]["deckCount"] = 28
    after_draw["current"]["result"] = 2
    after_draw["current"]["resultReason"] = "SimultaneousKnockout"
    events = [
        PromptChainStep(
            before=copy.deepcopy(observation),
            after=copy.deepcopy(after_attack),
            event_kind="attack",
            action=(0,),
            facts={"damage": 200},
        ),
        PromptChainStep(
            before=copy.deepcopy(after_attack),
            after=copy.deepcopy(after_nullifying),
            event_kind="nullifying_zero",
            forced=True,
            facts={},
        ),
        PromptChainStep(
            before=copy.deepcopy(after_nullifying),
            after=copy.deepcopy(after_ko),
            event_kind="knockout",
            forced=True,
            facts={
                "victimPlayerIndex": 1,
                "knockedOutCard": {"id": 401, "ex": True, "megaEx": True},
                "prizeYield": 3,
            },
        ),
        PromptChainStep(
            before=copy.deepcopy(after_ko),
            after=copy.deepcopy(after_promotion),
            event_kind="promotion",
            forced=True,
            facts={"playerIndex": 1},
        ),
        PromptChainStep(
            before=copy.deepcopy(after_promotion),
            after=copy.deepcopy(after_draw),
            event_kind="draw",
            forced=True,
            facts={"drawCount": 2, "forced": True},
        ),
    ]
    result = _compile_with_fixture(observation, _fixture_chain(observation, events=events))
    assert result["terminal"] == {
        "class": "draw",
        "mask": True,
        "reason": "terminalreasonsimultaneousknockout",
        "simultaneous_closeout_draw_preserved": True,
    }
    assert result["prompt_chain"]["event_count"] == 5
    assert result["prompt_chain"]["forced_event_count"] == 4
    assert result["immediate_effects"]["damage"] == {"value": 200.0, "mask": True}
    assert result["deck_out"]["forced_draw_count_before_next_strategic_decision"] == 2.0
    assert result["deck_out"]["acting_deck_before"] == 30
    assert result["deck_out"]["acting_deck_after"] == 28
    assert result["prize_yield"]["public_predicted_yield"] == 3
    utility = result["action_utility"]
    assert utility["values"][utility["layout"].index("opponent_knockout")] == 1.0
    assert utility["values"][utility["layout"].index("forced_promotion_count")] == 1.0
    assert result["selected_action"] == [0]

    # An extra selectable action after Nullifying Zero is not a forced prompt;
    # the compiler rejects it instead of crediting an invented choice.
    invalid = list(events)
    invalid[1] = PromptChainStep(
        before=copy.deepcopy(after_attack),
        after=copy.deepcopy(after_nullifying),
        event_kind="nullifying_zero",
        action=(1,),
        forced=True,
        facts={},
    )
    with pytest.raises(SimulatorRuleTargetError, match="may not invent another selected action"):
        _compile_with_fixture(observation, _fixture_chain(observation, events=invalid))


def test_hidden_state_serial_permutation_padding_and_unverified_masks_are_inert() -> None:
    observation = _observation(
        [
            {"type": "Attack", "attackId": 44},
            {"type": "Skill", "cardId": 77, "serial": 1001},
            {"type": "Skill", "cardId": 77, "serial": 1002},
        ]
    )
    chain = _fixture_chain(observation)
    first = _compile_with_fixture(observation, chain)
    variant = copy.deepcopy(observation)
    # All of these are private/replay-only aliases.  They cannot change the
    # public state fingerprint or give a logit/target a hidden-state bit.
    variant["search_begin_input"] = {"opponent_hand": [9901]}
    variant["logs"] = {"future": {"deck": [9902]}}
    variant["current"]["players"][0]["deckOrder"] = [1, 2, 3]
    variant["current"]["players"][1]["hand"] = [_card(999, 9009), _card(998, 9008)]
    variant["current"]["players"][1]["deck"] = [_card(997, 9007)]
    variant["current"]["players"][0]["bench"][0]["serial"] = 5001
    variant["select"]["option"][1]["serial"] = 5001
    variant_chain = _fixture_chain(variant)
    second = _compile_with_fixture(variant, variant_chain)
    assert public_rule_observation_fingerprint(observation) == public_rule_observation_fingerprint(variant)
    assert_public_target_invariance(first, second)

    # Option *row* order is execution alignment only.  A consistent selected
    # action remap preserves its semantic identity and must produce the same
    # public target facts without leaking the original row ordinal.
    permuted = copy.deepcopy(observation)
    permuted["select"]["option"] = list(reversed(permuted["select"]["option"]))
    permuted_result = _compile_with_fixture(
        permuted,
        _fixture_chain(permuted, selected=2),
        selected=2,
    )
    assert_public_target_invariance(first, permuted_result)

    vectors = rule_head_target_vectors(first)
    assert vectors["target_training_eligible"] is False
    for name in (
        "lethal_threat",
        "prize_race",
        "action_utility",
        "game_phase",
        "terminal_conversion",
        "turn_resources",
        "attack_readiness",
    ):
        assert not any(vectors[name]["mask"]), name
    assert vectors["action_utility"]["selected_option_indices"] == [0]



@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="zero-gate parity needs the optional torch sidecar",
)
def test_zero_gate_is_exact_baseline_logit_bypass_for_padded_legal_rows() -> None:
    """A disabled r298 route neither reads nor changes baseline logit bytes."""

    import torch

    # The r298 heads preserve arbitrary legal-option width and return the
    # exact same base-logit bytes when default-off.  This also guards against
    # padding/cap code accidentally reading an unwired residual.
    heads = R298RuleAuxiliaryHeads(R298RuleAuxHeadsConfig(d_model=4, route_width=5))
    base = torch.tensor([[1.0, -0.0, 3.0]], dtype=torch.float32)
    unchanged = heads.apply_to_policy(
        base,
        state_hidden=None,
        option_hidden=None,
        runtime_enabled=False,
    )
    assert unchanged is base
    assert unchanged.numpy().tobytes() == base.numpy().tobytes()
