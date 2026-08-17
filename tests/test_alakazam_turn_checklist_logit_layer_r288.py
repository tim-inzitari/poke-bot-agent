"""Contract tests for the bounded r288 Alakazam turn-checklist residual.

The layer is intentionally a deterministic, policy-side advisory.  These
tests exercise the public observable facts it may use, rather than relying on
hidden deck order, simulator state, or a learned checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from poke_bot import alakazam_new_list_heuristics as guide
from poke_bot import alakazam_turn_checklist_logit_layer as checklist_layer
from poke_bot.agent import PolicyAgent
from poke_bot.alakazam_turn_checklist_logit_layer import (
    CHANNEL_NAMES,
    apply_turn_checklist_logits,
    apply_turn_checklist_probabilities,
    evaluate_turn_checklist,
)


EXPECTED_CHANNEL_NAMES = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "replacement_alakazam_line",
    "unavoidable_draws_before_attack",
    "bench_prize_exposure",
    "immediate_disruption_outcome",
    "unknown_prize_robust_line",
    "terminal_before_forced_draw",
)

EXACT_LEDGER_DECK_FINGERPRINT = (
    "sha256:44284481e46dd2aac8d92bea417cbcfcda40db221d76fe39185a01d11754fce8"
)
OWN_DECK_LEDGER_SCHEMA = "poke_bot.own_deck_ledger/v1"


def _pokemon(
    card_id: int,
    hp: int = 100,
    *,
    energy: list[int] | None = None,
    rule_box: bool = False,
    pre_evolution: int | None = None,
    types: list[str] | None = None,
    mega_ex: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": card_id,
        "hp": hp,
        "maxHp": hp,
        "energyCards": [{"id": energy_id} for energy_id in (energy or [])],
        "tools": [],
    }
    if rule_box:
        value["ruleBox"] = True
    if pre_evolution is not None:
        value["preEvolution"] = pre_evolution
    if types is not None:
        value["types"] = list(types)
    if mega_ex:
        value["megaEx"] = True
    return value


def _player(
    *,
    hand: list[int] | None = None,
    hand_count: int | None = None,
    active: dict[str, Any] | None = None,
    bench: list[dict[str, Any]] | None = None,
    discard: list[int] | None = None,
    deck_count: int = 20,
    prizes: int = 6,
) -> dict[str, Any]:
    hand_cards = [{"id": card_id} for card_id in (hand or [])]
    return {
        "active": [] if active is None else [active],
        "bench": list(bench or []),
        "hand": hand_cards,
        "handCount": len(hand_cards) if hand_count is None else hand_count,
        "discard": [{"id": card_id} for card_id in (discard or [])],
        "deckCount": deck_count,
        "prize": [None] * prizes,
    }


def _obs(
    me: dict[str, Any],
    opponent: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    context: int = 0,
    effect_id: int | None = None,
    deck: list[dict[str, Any]] | None = None,
    stadium: list[dict[str, Any]] | None = None,
    min_count: int = 1,
    max_count: int = 1,
) -> dict[str, Any]:
    select: dict[str, Any] = {
        "context": context,
        "option": options,
        "minCount": min_count,
        "maxCount": max_count,
    }
    if effect_id is not None:
        select["effect"] = {"id": effect_id}
    if deck is not None:
        select["deck"] = deck
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, opponent],
            "stadium": list(stadium or []),
            "looking": [],
        },
        "select": select,
    }


def _trace_payload(trace: Any) -> dict[str, Any]:
    to_dict = getattr(trace, "to_dict", None)
    assert callable(to_dict), "checklist trace must provide an immutable audit payload"
    payload = to_dict()
    assert isinstance(payload, dict)
    return payload


def _channel(payload: dict[str, Any], name: str) -> dict[str, Any]:
    rows = payload["channels"]
    assert isinstance(rows, list)
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    raise AssertionError(f"missing checklist channel {name!r}")


def _raw(payload: dict[str, Any], name: str) -> list[float]:
    row = _channel(payload, name)
    values = row["raw"]
    assert isinstance(values, list)
    return [float(value) for value in values]


@dataclass(frozen=True)
class _Availability:
    lower: int
    upper: int
    expected: float | None = None
    probability_at_least_one: float | None = None
    exact: bool = False


def _ledger(
    lower: dict[int, int],
    *,
    upper_delta: int = 0,
    unknown_prize_slots: int | None = 1,
    integrity_ok: bool = True,
    fail_closed: bool = False,
    schema: str = OWN_DECK_LEDGER_SCHEMA,
    deck_fingerprint: str = EXACT_LEDGER_DECK_FINGERPRINT,
    actor: int | None = 0,
    fingerprint: str | None = "sha256:" + "0" * 64,
) -> Any:
    # The production core deliberately consumes only public, conservative
    # availability bounds.  A tiny structural stand-in keeps these tests free
    # of hidden deck/prize fixtures.
    availability = {
        card_id: _Availability(
            lower=count,
            upper=count + upper_delta,
            expected=float(count + upper_delta),
        )
        for card_id, count in lower.items()
    }
    return SimpleNamespace(
        schema=schema,
        deck_fingerprint=deck_fingerprint,
        actor=actor,
        fingerprint=fingerprint,
        availability_by_card=availability,
        unknown_prize_slots=unknown_prize_slots,
        integrity_ok=integrity_ok,
        fail_closed=fail_closed,
    )


def _evaluate(
    observation: dict[str, Any],
    candidates: list[list[int]],
    *,
    ledger_snapshot: Any = None,
) -> dict[str, Any]:
    trace = evaluate_turn_checklist(
        observation,
        candidates,
        guide.EXACT_DECK,
        ledger_snapshot=ledger_snapshot,
    )
    return _trace_payload(trace)


def test_exact_eight_channel_contract_and_auditable_shapes() -> None:
    me = _player(
        hand_count=11,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    opponent = _player(active=_pokemon(900, 201))
    observation = _obs(
        me,
        opponent,
        [
            {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ],
    )

    payload = _evaluate(observation, [[0], [1]])

    assert tuple(CHANNEL_NAMES) == EXPECTED_CHANNEL_NAMES
    assert tuple(payload["channel_names"]) == EXPECTED_CHANNEL_NAMES
    assert len(payload["channels"]) == len(EXPECTED_CHANNEL_NAMES)
    assert len(payload["residuals"]) == 2
    assert isinstance(payload["facts"], dict)
    assert isinstance(payload["guide_support"], dict)
    assert isinstance(payload["scalar_gates"], dict)
    for name in EXPECTED_CHANNEL_NAMES:
        row = _channel(payload, name)
        assert set(
            (
                "name",
                "raw",
                "normalized",
                "option_availability",
                "available",
                "reason",
                "applied_gate",
                "post_deduplication_signed_residual",
                "post_cap_residual",
                "group_winner",
            )
        ).issubset(row)
        assert len(row["raw"]) == 2
        assert len(row["normalized"]) == 2
        assert len(row["option_availability"]) == 2
        assert len(row["post_deduplication_signed_residual"]) == 2
        assert len(row["post_cap_residual"]) == 2
        assert len(row["group_winner"]) == 2
        assert isinstance(row["available"], bool)
        assert isinstance(row["reason"], str)
    # Calibration/Elmo diagnostics consume these normalized vectors directly,
    # but they must stay distinct from the raw evidence and its availability.
    assert set(payload["normalized_channel_vectors"]) == set(EXPECTED_CHANNEL_NAMES)
    assert set(payload["channel_status"]).issuperset(EXPECTED_CHANNEL_NAMES)

    # r293 records overlap decisions in the new layer itself.  The existing
    # learned routes are only described here; their scores are never edited.
    overlap = payload["channel_overlap_audit"]
    assert set(overlap) == set(EXPECTED_CHANNEL_NAMES)
    for name in EXPECTED_CHANNEL_NAMES:
        audit = overlap[name]
        assert set(
            (
                "existing_route_overlap_or_distinct_reason",
                "attenuation_or_suppression_decision",
                "applied_gate",
                "overlap_group",
                "group_winner",
                "gated_signed_residual",
                "post_deduplication_signed_residual",
                "post_total_cap_signed_residual",
            )
        ).issubset(audit)
        assert len(audit["group_winner"]) == 2
        assert len(audit["post_deduplication_signed_residual"]) == 2
        assert len(audit["post_total_cap_signed_residual"]) == 2
    assert overlap["bench_prize_exposure"]["applied_gate"] == 0.0
    assert overlap["immediate_disruption_outcome"]["applied_gate"] == 0.0
    assert payload["guide_support_trace_only"] is True
    assert payload["guide_support_runtime_residual"] == 0.0


def test_ko_math_uses_ceiling_hp_over_twenty_and_safe_spend_is_post_cost() -> None:
    # 201 remaining HP requires 11 cards, not ten.  The only non-attack action
    # spends one card from a 10-card hand and therefore cannot be advertised as
    # a safe spend above the attack threshold.
    me = _player(
        hand=[guide.BATTLE_CAGE],
        hand_count=10,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    opponent = _player(active=_pokemon(900, 201))
    observation = _obs(
        me,
        opponent,
        [
            {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
            {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ],
    )
    payload = _evaluate(observation, [[0], [1], [2]])

    ko = _channel(payload, "ko_hand_threshold")
    assert ko["available"]
    # The exposed fact remains inspectable even if no candidate is currently
    # lethal.  Keep the assertion deliberately schema-facing, not a hidden
    # model implementation detail.
    assert any(value == 11 for value in payload["facts"].values())
    safe = _raw(payload, "safe_spend_above_threshold")
    assert safe[0] < 0.0
    assert _raw(payload, "ko_hand_threshold")[1] <= 0.0


def test_replacement_line_bench_choice_and_forced_draw_accounting() -> None:
    # A valid Poffin setup selection for an Abra is not yet a live replacement
    # line; it is a later, unresolved route.  Fez cannot be selected by
    # Poffin (its HP is too high), so prize exposure is covered separately as
    # a direct, legal bench play below.
    me = _player(
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=4,
    )
    opponent = _player(active=_pokemon(900, 200))
    setup = _obs(
        me,
        opponent,
        [
            {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
            {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
        ],
        context=2,
        effect_id=guide.BUDDY_BUDDY_POFFIN,
        deck=[{"id": guide.ABRA}, {"id": guide.DUNSPARCE}],
    )
    setup_payload = _evaluate(setup, [[0], [1]])
    replacement = _raw(setup_payload, "replacement_alakazam_line")
    assert not _channel(setup_payload, "replacement_alakazam_line")["available"]
    assert replacement == [0.0, 0.0]

    direct_fez = _evaluate(
        _obs(
            _player(
                hand=[guide.FEZANDIPITI_EX],
                active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
            ),
            opponent,
            [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    exposure = _raw(direct_fez, "bench_prize_exposure")
    assert exposure[0] < exposure[1]

    # Enriching Energy is an immediate forced draw of four.  Telepath is an
    # energy/search card, not a draw, so it must not inherit that debit.
    draw_obs = _obs(
        _player(
            hand=[guide.ENRICHING_ENERGY, guide.TELEPATH_PSYCHIC_ENERGY],
            active=_pokemon(guide.DUNSPARCE, 70),
            deck_count=4,
        ),
        opponent,
        [
            {"type": 8, "area": 2, "index": 0, "playerIndex": 0, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 8, "area": 2, "index": 1, "playerIndex": 0, "inPlayArea": 4, "inPlayIndex": 0},
        ],
    )
    draw_payload = _evaluate(draw_obs, [[0], [1]])
    draws = _raw(draw_payload, "unavoidable_draws_before_attack")
    assert draws[0] < draws[1]
    assert any(value == 4 for value in draw_payload["facts"].values())


def test_disruption_is_immediate_and_terminal_line_ends_before_zero_deck_draw() -> None:
    me = _player(
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        hand_count=5,
        deck_count=0,
        prizes=1,
    )
    opponent = _player(
        active=_pokemon(900, 100, energy=[guide.MIST_ENERGY, guide.PSYCHIC_ENERGY]),
    )

    hammer = _obs(
        me,
        opponent,
        [
            {"type": 5, "area": 4, "index": 0, "playerIndex": 1, "energyIndex": 0},
            {"type": 5, "area": 4, "index": 0, "playerIndex": 1, "energyIndex": 1},
        ],
        context=30,
        effect_id=guide.ENHANCED_HAMMER,
    )
    hammer_payload = _evaluate(hammer, [[0], [1]])
    disruption = _raw(hammer_payload, "immediate_disruption_outcome")
    assert disruption[0] > disruption[1]

    terminal = _obs(
        me,
        _player(active=_pokemon(901, 100)),
        [
            {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ],
    )
    terminal_payload = _evaluate(terminal, [[0], [1]])
    terminal_raw = _raw(terminal_payload, "terminal_before_forced_draw")
    assert terminal_raw[0] > terminal_raw[1]
    assert terminal_payload["facts"]


def test_unknown_prize_route_uses_only_ledger_lower_bounds() -> None:
    me = _player(
        hand=[guide.POKE_PAD],
        active=_pokemon(900, 100),
        bench=[_pokemon(guide.ABRA, 60)],
    )
    opponent = _player(active=_pokemon(900, 240))
    observation = _obs(
        me,
        opponent,
        [
            {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
            {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
        ],
        effect_id=guide.POKE_PAD,
        deck=[{"id": guide.KADABRA}, {"id": guide.FEZANDIPITI_EX}],
    )
    candidates = [[0], [1]]

    robust = _evaluate(
        observation,
        candidates,
        ledger_snapshot=_ledger(
            {
                guide.KADABRA: 1,
                guide.ALAKAZAM: 1,
                guide.PSYCHIC_ENERGY: 1,
            },
            upper_delta=0,
        ),
    )
    upper_only_change = _evaluate(
        observation,
        candidates,
        ledger_snapshot=_ledger(
            {
                guide.KADABRA: 1,
                guide.ALAKAZAM: 1,
                guide.PSYCHIC_ENERGY: 1,
            },
            upper_delta=3,
        ),
    )
    brittle = _evaluate(
        observation,
        candidates,
        ledger_snapshot=_ledger(
            {
                guide.KADABRA: 0,
                guide.ALAKAZAM: 0,
                guide.PSYCHIC_ENERGY: 0,
            },
            upper_delta=3,
        ),
    )

    assert _channel(robust, "unknown_prize_robust_line")["available"]
    assert _raw(robust, "unknown_prize_robust_line") == pytest.approx(
        _raw(upper_only_change, "unknown_prize_robust_line")
    )
    assert _raw(robust, "unknown_prize_robust_line") != pytest.approx(
        _raw(brittle, "unknown_prize_robust_line")
    )


def test_unknown_prize_paths_use_visible_resources_and_exact_alternatives() -> None:
    """Q7 names the actual natural/Candy dependencies, not all line cards.

    A card already visible in our hand is physically available even when the
    ledger has no draw-pile lower bound for it.  The two routes have different
    required resources: natural evolution uses Kadabra; the Candy route does
    not.  This is a public worst-case resource proof, never a hidden-prize
    prediction.
    """

    opponent = _player(active=_pokemon(900, 200))

    def evaluate_path(hand: list[int], ledger_snapshot: Any) -> dict[str, Any]:
        return _evaluate(
            _obs(
                _player(hand=hand, active=_pokemon(901, 100)),
                opponent,
                [{"type": 14}],
            ),
            [[0]],
            ledger_snapshot=ledger_snapshot,
        )

    natural = evaluate_path(
        [guide.ABRA, guide.KADABRA, guide.ALAKAZAM, guide.PSYCHIC_ENERGY],
        _ledger({}, unknown_prize_slots=2),
    )
    natural_facts = natural["facts"]
    assert natural_facts["unknown_prize_slots"] == 2
    assert natural_facts["unknown_prize_route_classification"] == "natural"
    natural_path = natural_facts["safe_paths"]["natural"]
    assert natural_path["lower_bound_proven"] is True
    assert set(natural_path["required_resources"]) == {
        guide.ABRA,
        guide.KADABRA,
        guide.ALAKAZAM,
        guide.PSYCHIC_ENERGY,
    }
    assert isinstance(natural_facts["key_lower_bounds"], dict)
    assert natural_facts["unknown_prize_unavailable_or_brittle_reason"] is None

    candy = evaluate_path(
        [guide.ABRA, guide.RARE_CANDY, guide.ALAKAZAM, guide.TELEPATH_PSYCHIC_ENERGY],
        _ledger({}, unknown_prize_slots=2),
    )
    candy_facts = candy["facts"]
    assert candy_facts["unknown_prize_route_classification"] == "rare_candy"
    candy_path = candy_facts["safe_paths"]["rare_candy"]
    assert candy_path["lower_bound_proven"] is True
    assert set(candy_path["required_resources"]) == {
        guide.ABRA,
        guide.RARE_CANDY,
        guide.ALAKAZAM,
        guide.TELEPATH_PSYCHIC_ENERGY,
    }
    assert guide.KADABRA not in candy_path["required_resources"]


def test_unknown_prize_channel_fails_closed_for_unknown_slots_or_bad_ledger_binding() -> None:
    """Q7 requires an exact current own-deck ledger, not a loose estimate."""

    observation = _obs(
        _player(
            hand=[guide.ABRA, guide.KADABRA, guide.ALAKAZAM, guide.PSYCHIC_ENERGY],
            active=_pokemon(901, 100),
        ),
        _player(active=_pokemon(900, 200)),
        [{"type": 14}],
    )

    for ledger_snapshot in (
        _ledger({}, unknown_prize_slots=0),
        _ledger({}, unknown_prize_slots=None),
        _ledger({}, schema="wrong-ledger-schema"),
        _ledger({}, deck_fingerprint="sha256:" + "f" * 64),
        _ledger({}, actor=1),
        _ledger({}, fingerprint=None),
    ):
        payload = _evaluate(observation, [[0]], ledger_snapshot=ledger_snapshot)
        robust = _channel(payload, "unknown_prize_robust_line")
        assert not robust["available"]
        assert robust["raw"] == [0.0]
        facts = payload["facts"]
        assert facts["unknown_prize_unavailable_or_brittle_reason"]


def test_residual_is_bounded_permutation_equivariant_and_neutral_when_unsafe() -> None:
    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=1,
    )
    opponent = _player(active=_pokemon(900, 100))
    observation = _obs(
        me,
        opponent,
        [
            {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ],
    )
    logits = torch.tensor([0.25, -0.25])
    logits_before = logits.clone()
    adjusted, trace = apply_turn_checklist_logits(
        observation,
        [[0], [1]],
        guide.EXACT_DECK,
        logits=logits,
    )
    values = torch.as_tensor(adjusted, dtype=torch.float32)
    # The policy-side residual creates a new score vector; it never mutates a
    # checkpoint tensor or its caller-owned tensor in place.
    assert torch.equal(logits, logits_before)
    assert torch.max(torch.abs(values - logits)) <= 0.100001
    payload = _trace_payload(trace)
    assert max(abs(float(value)) for value in payload["residuals"]) <= 0.100001

    permuted, permuted_trace = apply_turn_checklist_logits(
        observation,
        [[1], [0]],
        guide.EXACT_DECK,
        logits=logits.flip(0),
    )
    assert torch.allclose(torch.as_tensor(permuted).flip(0), values, atol=1e-6)
    assert _trace_payload(permuted_trace)["residuals"][::-1] == pytest.approx(
        payload["residuals"]
    )

    wrong_deck, wrong_trace = apply_turn_checklist_logits(
        observation,
        [[0], [1]],
        [1] * 60,
        logits=logits,
    )
    assert torch.equal(torch.as_tensor(wrong_deck), logits)
    wrong_payload = _trace_payload(wrong_trace)
    assert not wrong_payload["available"]
    assert wrong_payload["residuals"] == [0.0, 0.0]

    malformed_trace = evaluate_turn_checklist(
        {"select": {"option": []}}, [[0]], guide.EXACT_DECK
    )
    malformed_payload = _trace_payload(malformed_trace)
    assert not malformed_payload["available"]
    assert malformed_payload["residuals"] == [0.0]

    probabilities, probability_trace = apply_turn_checklist_probabilities(
        observation,
        [[0], [1]],
        guide.EXACT_DECK,
        probabilities=[0.5, 0.5],
    )
    assert sum(float(value) for value in probabilities) == pytest.approx(1.0)
    assert _trace_payload(probability_trace)["residuals"] == pytest.approx(
        payload["residuals"]
    )


def test_policy_agent_gate_is_default_off_even_for_the_exact_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POKEBOT_ALAKAZAM_TURN_CHECKLIST_LOGIT_LAYER", raising=False)
    policy = PolicyAgent(model=None, deck=list(guide.EXACT_DECK))

    assert policy.turn_checklist_logit_layer_enabled is False
    assert policy.last_turn_checklist_logit_trace is None


def test_replacement_line_ready_fact_requires_visible_benched_powered_alakazam() -> None:
    """Do not confuse the current attacker with a next attacker route.

    In particular, an Active Alakazam with Psychic energy and an empty Bench is
    not a replacement line.  A visibly powered Bench Alakazam is.  Every
    partial evolution shape remains ``timing_unresolved`` unless a real legal
    transition has already made it a ready Bench attacker.
    """

    opponent = _player(active=_pokemon(900, 200))
    options = [{"type": 14}, {"type": 14}]
    cases = (
        (
            _player(
                active=_pokemon(
                    guide.ALAKAZAM,
                    140,
                    energy=[guide.PSYCHIC_ENERGY],
                ),
            ),
            False,
            False,
            False,
            "not_live",
        ),
        (
            _player(
                active=_pokemon(900, 100),
                bench=[
                    _pokemon(
                        guide.ALAKAZAM,
                        140,
                        energy=[guide.PSYCHIC_ENERGY],
                    )
                ],
            ),
            True,
            False,
            False,
            "ready",
        ),
        (
            _player(
                hand=[guide.ALAKAZAM],
                active=_pokemon(900, 100),
                bench=[
                    _pokemon(
                        guide.KADABRA,
                        80,
                        energy=[guide.PSYCHIC_ENERGY],
                        pre_evolution=guide.ABRA,
                    )
                ],
            ),
            False,
            False,
            True,
            "not_live",
        ),
        (
            _player(
                hand=[guide.RARE_CANDY, guide.ALAKAZAM],
                active=_pokemon(900, 100),
                bench=[_pokemon(guide.ABRA, 60)],
            ),
            False,
            False,
            True,
            "not_live",
        ),
    )

    for me, expected_ready, expected_completable, expected_unresolved, expected_status in cases:
        payload = _evaluate(_obs(me, opponent, options), [[0], [1]])
        assert payload["facts"]["next_alakazam_line_ready"] is expected_ready
        assert (
            payload["facts"]["next_alakazam_line_completable"]
            is expected_completable
        )
        assert (
            payload["facts"]["next_alakazam_line_visible_but_timing_unresolved"]
            is expected_unresolved
        )
        assert payload["facts"]["next_alakazam_line_status"] == expected_status

    # This is the exact public exception: the option names a Bench Kadabra,
    # evolves it with a visible Alakazam from hand, and the target already has
    # Psychic energy.  That is completable, but not a ready line until the
    # legal transition resolves.
    completable = _evaluate(
        _obs(
            _player(
                hand=[guide.ALAKAZAM],
                active=_pokemon(900, 100),
                bench=[
                    _pokemon(
                        guide.KADABRA,
                        80,
                        energy=[guide.PSYCHIC_ENERGY],
                        pre_evolution=guide.ABRA,
                    )
                ],
            ),
            opponent,
            [
                {
                    "type": 9,
                    "area": 2,
                    "index": 0,
                    "playerIndex": 0,
                    "inPlayArea": 5,
                    "inPlayIndex": 0,
                    "inPlayPlayerIndex": 0,
                },
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    assert completable["facts"]["next_alakazam_line_ready"] is False
    assert completable["facts"]["next_alakazam_line_completable"] is True
    assert completable["facts"]["next_alakazam_line_status"] == "completable"
    assert _raw(completable, "replacement_alakazam_line")[0] > _raw(
        completable, "replacement_alakazam_line"
    )[1]


def test_unknown_prize_or_draw_availability_never_becomes_a_live_replacement() -> None:
    """Even a strong ledger lower bound is not an already-live Bench line."""

    me = _player(active=_pokemon(900, 100), hand=[guide.POKE_PAD])
    opponent = _player(active=_pokemon(901, 100))
    payload = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
        ),
        [[0], [1]],
        ledger_snapshot=_ledger(
            {
                guide.ABRA: 3,
                guide.KADABRA: 3,
                guide.ALAKAZAM: 2,
                guide.PSYCHIC_ENERGY: 1,
            },
            upper_delta=1,
            unknown_prize_slots=6,
        ),
    )

    assert payload["facts"]["next_alakazam_line_ready"] is False
    assert payload["facts"]["next_alakazam_line_status"] == "not_live"
    # The ledger may answer the separate robust-line question, but it must not
    # leak an inferred draw/prize outcome into the replacement-line fact.  No
    # option at this factorized stage completes that route, so Q7 carries its
    # public proof in facts/masks while remaining a flat neutral residual.
    robust = _channel(payload, "unknown_prize_robust_line")
    assert not robust["available"]
    assert robust["option_availability"] == [True, True]
    assert robust["raw"] == [0.0, 0.0]
    assert payload["facts"]["unknown_prize_route_classification"] == "natural"
    assert not _channel(payload, "replacement_alakazam_line")["available"]


def test_replacement_trace_has_required_bench_only_r292_facts() -> None:
    """Q3 always explains its Bench-only classification as public trace data."""

    opponent = _player(active=_pokemon(900, 200))
    active_only = _evaluate(
        _obs(
            _player(
                active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY])
            ),
            opponent,
            [{"type": 14}],
        ),
        [[0]],
    )
    facts = active_only["facts"]
    assert facts["bench_only"] is True
    assert facts["classification"] == "not_live"
    assert isinstance(facts["visible_required_resources"], (list, dict))
    assert isinstance(facts["unavailable_or_not_live_reason"], str)
    assert not _channel(active_only, "replacement_alakazam_line")["available"]

    ready = _evaluate(
        _obs(
            _player(
                active=_pokemon(900, 100),
                bench=[
                    _pokemon(
                        guide.ALAKAZAM,
                        140,
                        energy=[guide.PSYCHIC_ENERGY],
                    )
                ],
            ),
            opponent,
            [{"type": 14}],
        ),
        [[0]],
    )
    ready_facts = ready["facts"]
    assert ready_facts["bench_only"] is True
    assert ready_facts["classification"] == "ready"
    assert isinstance(ready_facts["unavailable_or_not_live_reason"], str)


def test_mist_and_rock_fighting_protection_do_not_share_a_universal_rule() -> None:
    """Mist is absolute; Rock Fighting needs a visible Fighting target type."""

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    options = [
        {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
        {"type": 14},
    ]

    mist = _evaluate(
        _obs(me, _player(active=_pokemon(900, 100, energy=[guide.MIST_ENERGY])), options),
        [[0], [1]],
    )
    assert mist["facts"]["powerful_hand_effect_prevented"] is True
    assert mist["facts"]["powerful_hand_conditional_protection_unknown"] is False
    assert not _channel(mist, "ko_hand_threshold")["available"]
    assert _raw(mist, "ko_hand_threshold") == [0.0, 0.0]

    ambiguous_rock = _evaluate(
        _obs(
            me,
            _player(active=_pokemon(900, 100, energy=[guide.ROCK_FIGHTING_ENERGY])),
            options,
        ),
        [[0], [1]],
    )
    assert ambiguous_rock["facts"]["powerful_hand_effect_prevented"] is False
    assert ambiguous_rock["facts"]["powerful_hand_conditional_protection_unknown"] is True
    assert not _channel(ambiguous_rock, "ko_hand_threshold")["available"]

    non_fighting_rock = _evaluate(
        _obs(
            me,
            _player(
                active=_pokemon(
                    900,
                    100,
                    energy=[guide.ROCK_FIGHTING_ENERGY],
                    types=["Lightning"],
                )
            ),
            options,
        ),
        [[0], [1]],
    )
    assert non_fighting_rock["facts"]["powerful_hand_effect_prevented"] is False
    assert non_fighting_rock["facts"]["powerful_hand_conditional_protection_unknown"] is False
    assert _raw(non_fighting_rock, "ko_hand_threshold")[0] > 0.0

    fighting_rock = _evaluate(
        _obs(
            me,
            _player(
                active=_pokemon(
                    900,
                    100,
                    energy=[guide.ROCK_FIGHTING_ENERGY],
                    types=["Fighting"],
                )
            ),
            options,
        ),
        [[0], [1]],
    )
    assert fighting_rock["facts"]["powerful_hand_effect_prevented"] is True
    assert not _channel(fighting_rock, "ko_hand_threshold")["available"]


def test_boss_post_cost_and_optional_evolution_draws_are_causally_separate() -> None:
    """Boss selection sees the already-paid cost; evolution does not force draw."""

    me = _player(
        hand_count=5,  # Public hand *after* Boss's one-card cost.
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        prizes=1,
    )
    opponent = _player(
        active=_pokemon(900, 200),
        bench=[
            _pokemon(901, 100, rule_box=True),
            _pokemon(902, 120),
        ],
    )
    boss_target = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 3, "area": 5, "index": 0, "playerIndex": 1},
                {"type": 3, "area": 5, "index": 1, "playerIndex": 1},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    assert _raw(boss_target, "immediate_disruption_outcome")[0] > _raw(
        boss_target, "immediate_disruption_outcome"
    )[1]

    # At the main-stage prefix, a legal Boss card has not yet selected a
    # target.  It cannot claim an immediate prize or retreat result.
    boss_prefix = _evaluate(
        _obs(
            _player(
                hand=[guide.BOSS_ORDERS],
                hand_count=6,
                active=_pokemon(
                    guide.ALAKAZAM,
                    140,
                    energy=[guide.PSYCHIC_ENERGY],
                ),
            ),
            opponent,
            [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    assert not _channel(boss_prefix, "immediate_disruption_outcome")["available"]

    # Evolving can expose an optional draw prompt later, but this action alone
    # must not debit Kadabra/Alakazam/Dudunsparce cards immediately.
    evolve = _evaluate(
        _obs(
            _player(
                hand=[guide.KADABRA],
                active=_pokemon(guide.ABRA, 60),
                deck_count=2,
            ),
            _player(active=_pokemon(903, 100)),
            [
                {"type": 9, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    assert not _channel(evolve, "unavoidable_draws_before_attack")["available"]
    assert evolve["facts"]["maximum_exact_forced_draw_count"] == 0


def test_terminal_channel_requires_visible_prize_yield_not_an_inferred_outcome() -> None:
    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=1,
    )
    # Remaining HP alone is insufficient if the target has no visible card
    # identity/prize yield.  The layer must not call this a terminal line.
    opponent = _player(
        active={"hp": 100, "maxHp": 100, "energyCards": [], "tools": []}
    )
    payload = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )

    assert not _channel(payload, "terminal_before_forced_draw")["available"]
    assert _raw(payload, "terminal_before_forced_draw") == [0.0, 0.0]


def test_visible_generic_attack_effect_prevention_masks_ko_and_terminal() -> None:
    """Not every protection effect is represented by Mist/Rock Energy.

    The exact visible Skeledirge (203) attack-effect prevention case is a
    public blocker.  It must neutralize the Powerful Hand KO/terminal claims,
    rather than leave an optimistic residual based on HP and hand size alone.
    """

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=1,
    )
    skeledirge = _player(active=_pokemon(203, 100))
    payload = _evaluate(
        _obs(
            me,
            skeledirge,
            [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )

    assert not _channel(payload, "ko_hand_threshold")["available"]
    assert not _channel(payload, "safe_spend_above_threshold")["available"]
    assert not _channel(payload, "terminal_before_forced_draw")["available"]
    assert _raw(payload, "ko_hand_threshold") == [0.0, 0.0]
    assert _raw(payload, "terminal_before_forced_draw") == [0.0, 0.0]


def test_mega_ex_terminal_prize_yield_is_three_not_two() -> None:
    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=3,
    )
    opponent = _player(active=_pokemon(950, 100, mega_ex=True))
    payload = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )

    assert _channel(payload, "terminal_before_forced_draw")["available"]
    assert _raw(payload, "terminal_before_forced_draw")[0] > 0.0


def test_raw_replay_prize_yield_uses_card_identity_without_rulebox_flags() -> None:
    """Actual replay rows are sparse: known prize cards carry only their id.

    The exact raw identities cover ordinary ex (121), Fez (140), and Mega ex
    (652); fixture-only rule-box flags must not be required to close prizes.
    """

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=2,
    )
    for two_prize_id in (121, guide.FEZANDIPITI_EX):
        payload = _evaluate(
            _obs(
                me,
                _player(active=_pokemon(two_prize_id, 100)),
                [
                    {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                    {"type": 14},
                ],
            ),
            [[0], [1]],
        )
        assert _raw(payload, "terminal_before_forced_draw")[0] > 0.0

    mega_me = dict(me)
    mega_me["prize"] = [None] * 3
    # Raw card identity 652 is a known three-prize Mega.  It intentionally
    # omits fixture-only ``megaEx`` / ``ruleBox`` fields.
    raw_mega = _player(active=_pokemon(652, 100))
    mega_payload = _evaluate(
        _obs(
            mega_me,
            raw_mega,
            [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    assert _raw(mega_payload, "terminal_before_forced_draw")[0] > 0.0


def test_string_enum_options_follow_the_same_public_legal_stage() -> None:
    """Replay JSON may serialize native option enums by name rather than int."""

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=1,
    )
    opponent = _player(active=_pokemon(900, 100))
    numeric = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    named = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": "Attack", "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": "End"},
            ],
        ),
        [[0], [1]],
    )
    assert _raw(named, "ko_hand_threshold") == pytest.approx(
        _raw(numeric, "ko_hand_threshold")
    )
    assert _raw(named, "terminal_before_forced_draw") == pytest.approx(
        _raw(numeric, "terminal_before_forced_draw")
    )


def test_string_enum_card_and_bench_target_match_numeric_boss_stage() -> None:
    """Named replay enums must retain the exact target-zone binding."""

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        prizes=2,
    )
    opponent = _player(
        active=_pokemon(900, 100),
        bench=[_pokemon(guide.FEZANDIPITI_EX, 100)],
    )
    numeric = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 3, "area": 5, "index": 0, "playerIndex": 1},
                {"type": 14},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    named = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": "Card", "area": "Bench", "index": 0, "playerIndex": 1},
                {"type": "End"},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    assert _raw(named, "immediate_disruption_outcome") == pytest.approx(
        _raw(numeric, "immediate_disruption_outcome")
    )


def test_unresolved_search_prefix_is_neutral_safe_spend_not_a_false_negative() -> None:
    me = _player(
        hand=[guide.RARE_CANDY],
        hand_count=10,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    opponent = _player(active=_pokemon(900, 200))
    payload = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1], [2]],
    )
    safe = _channel(payload, "safe_spend_above_threshold")
    assert safe["raw"] == [0.0, 0.0, 0.0]
    assert safe["option_availability"] == [False, True, True]


def test_default_guide_gate_is_exact_zero_and_cannot_change_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r293 keeps broad historic guide scores out of the default residual."""

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=1,
    )
    opponent = _player(active=_pokemon(900, 100))
    observation = _obs(
        me,
        opponent,
        [
            {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
            {"type": 14},
        ],
    )
    # Make an intentionally extreme historic score.  It is distinct enough to
    # reveal accidental use, but the default/staged path has no authority to
    # incorporate or cancel it.
    monkeypatch.setattr(
        checklist_layer._guide,
        "guide_scores",
        lambda *_args, **_kwargs: [99.0, -99.0],
    )
    without_guide = _trace_payload(
        evaluate_turn_checklist(
            observation,
            [[0], [1]],
            guide.EXACT_DECK,
            config={
                "schema": "poke_bot.alakazam_turn_checklist_heuristic_logit_layer_config/v1",
                "runtime": {
                    "total_residual_cap": 0.1,
                    "scalar_gates": {"separate_guide_gate": 0.0},
                },
            },
        )
    )
    default = _evaluate(observation, [[0], [1]])

    assert default["scalar_gates"]["separate_guide_gate"] == 0.0
    assert without_guide["scalar_gates"]["separate_guide_gate"] == 0.0
    # The trace may remain zero/unavailable or trace-only in a future audit,
    # but it has exactly no default residual authority either way.
    assert without_guide["residuals"] == pytest.approx(default["residuals"])
    assert set(without_guide["normalized_channel_vectors"]) == set(EXPECTED_CHANNEL_NAMES)

    # A nonzero scalar in an ordinary staged mapping is still fail-closed.
    # Only a future, explicitly receipted experiment may exercise the broad
    # guide path; it must never be smuggled into this default contract.
    unauthorized = _trace_payload(
        evaluate_turn_checklist(
            observation,
            [[0], [1]],
            guide.EXACT_DECK,
            config={
                "schema": "poke_bot.alakazam_turn_checklist_heuristic_logit_layer_config/v1",
                "runtime": {
                    "total_residual_cap": 0.1,
                    "scalar_gates": {"separate_guide_gate": 0.10},
                },
            },
        )
    )
    assert unauthorized["scalar_gates"]["separate_guide_gate"] == 0.0
    assert unauthorized["residuals"] == pytest.approx(default["residuals"])


def test_hilda_dawn_complete_multiselect_is_neutral_not_false_unsafe_spend() -> None:
    """A completed tutor selection must not inherit an invented -1 supporter.

    Hilda/Dawn were already paid before their deck-selection prompt.  Their
    multi-select contents are known cards, not a factorized prefix with an
    unknown later resolution.  The threshold channel can be neutral if no
    exact distinction is available; it may not call this completed selection
    an unsafe card spend.
    """

    for supporter in (guide.HILDA, guide.DAWN):
        me = _player(
            hand_count=10,
            active=_pokemon(
                guide.ALAKAZAM,
                140,
                energy=[guide.PSYCHIC_ENERGY],
            ),
        )
        opponent = _player(active=_pokemon(900, 200))
        payload = _evaluate(
            _obs(
                me,
                opponent,
                [
                    {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
                    {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
                ],
                effect_id=supporter,
                deck=[{"id": guide.ALAKAZAM}, {"id": guide.PSYCHIC_ENERGY}],
            ),
            [[0, 1]],
        )
        safe = _channel(payload, "safe_spend_above_threshold")
        assert not safe["available"] or safe["raw"] == [0.0]
        assert safe["raw"] != [-1.0]


def test_dudunsparce_selected_draw_is_exact_once_and_not_double_counted() -> None:
    """A resolved Run Away Draw selection is an exact three-card draw.

    If a compatibility surface also exposes a Yes prompt in the same complete
    candidate, it names the same trigger rather than six forced cards.
    """

    me = _player(active=_pokemon(guide.DUDUNSPARCE, 70), deck_count=3)
    opponent = _player(active=_pokemon(900, 100))
    payload = _evaluate(
        _obs(
            me,
            opponent,
            [
                {"type": 10, "area": 4, "index": 0, "playerIndex": 0},
                {"type": 1},
                {"type": 14},
            ],
            effect_id=guide.DUDUNSPARCE,
        ),
        [[0, 1], [2]],
    )

    draws = _channel(payload, "unavoidable_draws_before_attack")
    assert payload["facts"]["maximum_exact_forced_draw_count"] == 3
    # Run Away Draw returns Dudunsparce (and its attached cards) to the deck
    # before the three cards are drawn.  It is still an exact three-card
    # mandatory draw, but not a false deck-empty line at deckCount=3.
    assert draws["raw"][0] > -1.0
    assert draws["raw"][0] <= draws["raw"][1]


def test_optional_stop_empty_candidate_is_a_legal_neutral_comparator() -> None:
    """An optional factorised selection can offer STOP as the empty row."""

    observation = _obs(
        _player(
            hand_count=5,
            active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        ),
        _player(active=_pokemon(900, 100)),
        [
            {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
        ],
        min_count=0,
        max_count=1,
    )
    payload = _evaluate(observation, [[], [0]])

    assert payload["reason"] != "candidate_option_alignment_invalid"
    assert payload["available"]
    ko = _channel(payload, "ko_hand_threshold")
    assert ko["option_availability"] == [True, True]
    assert ko["raw"][1] > ko["raw"][0]


def test_remote_neutral_probability_path_preserves_exact_zero_entries() -> None:
    """A fail-closed remote prior must be returned byte-for-value unchanged."""

    probabilities = [0.0, 0.25, 0.75]
    output, trace = apply_turn_checklist_probabilities(
        {"select": {"option": []}},
        [[0], [1], [2]],
        [1] * 60,
        probabilities=probabilities,
    )

    assert output == probabilities
    assert output[0] == 0.0
    payload = _trace_payload(trace)
    assert not payload["available"]
    assert payload["residuals"] == [0.0, 0.0, 0.0]


def test_factorized_common_prefix_does_not_recredit_prior_forced_draw_or_cost() -> None:
    """Only the new suffix of a factorised candidate can affect this turn.

    The shared Enriching attachment was already selected at the prior stage.
    Re-seeing it in both full candidate encodings must not charge a second
    four-card forced draw or re-credit its hand delta at the next suffix.
    """

    observation = _obs(
        _player(
            hand=[guide.ENRICHING_ENERGY],
            hand_count=5,
            active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
            deck_count=4,
        ),
        _player(active=_pokemon(900, 100)),
        [
            {
                "type": 8,
                "area": 2,
                "index": 0,
                "playerIndex": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            },
            {"type": 14},
        ],
    )
    payload = _evaluate(observation, [[0], [0, 1]])

    draws = _channel(payload, "unavoidable_draws_before_attack")
    assert payload["facts"]["maximum_exact_forced_draw_count"] == 0
    assert draws["raw"] == [0.0, 0.0]
    assert not draws["available"]
    safe = _channel(payload, "safe_spend_above_threshold")
    assert safe["raw"] == [0.0, 0.0]


def test_hilda_dawn_main_stage_prefix_is_unresolved_not_false_unsafe_spend() -> None:
    """A supporter play before its required tutor resolution has no exact cost line."""

    for supporter in (guide.HILDA, guide.DAWN):
        observation = _obs(
            _player(
                hand=[supporter],
                hand_count=10,
                active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
            ),
            _player(active=_pokemon(900, 200)),
            [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        )
        payload = _evaluate(observation, [[0], [1], [2]])
        safe = _channel(payload, "safe_spend_above_threshold")
        assert safe["raw"] == [0.0, 0.0, 0.0]
        assert safe["option_availability"] == [False, True, True]


def test_known_attack_effect_blockers_mask_ko_safe_spend_and_terminal() -> None:
    """Every verified public blocker must neutralize the closure claims."""

    me = _player(
        hand_count=5,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        deck_count=0,
        prizes=1,
    )
    for blocker_id in (203, 835, 1136):
        payload = _evaluate(
            _obs(
                me,
                _player(active=_pokemon(blocker_id, 100)),
                [
                    {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                    {"type": 14},
                ],
            ),
            [[0], [1]],
        )
        for name in (
            "ko_hand_threshold",
            "safe_spend_above_threshold",
            "terminal_before_forced_draw",
        ):
            assert not _channel(payload, name)["available"]
            assert _raw(payload, name) == [0.0, 0.0]


def test_boss_requires_an_explicit_opponent_bench_target_and_current_active() -> None:
    """Do not score forged Boss targets or a prize map with no Active card."""

    me = _player(
        hand_count=5,  # The public target prompt is after the Boss cost.
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
        bench=[_pokemon(guide.FEZANDIPITI_EX, 100)],
        prizes=2,
    )
    ordinary_active = _player(active=_pokemon(900, 100), bench=[_pokemon(901, 100)])

    own_bench_target = _evaluate(
        _obs(
            me,
            ordinary_active,
            [
                {"type": 3, "area": 5, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    own_bench = _channel(own_bench_target, "immediate_disruption_outcome")
    assert own_bench["raw"] == [0.0, 0.0]
    assert not own_bench["available"]

    opponent_active_fez = _evaluate(
        _obs(
            me,
            _player(active=_pokemon(guide.FEZANDIPITI_EX, 100)),
            [
                {"type": 3, "area": 4, "index": 0, "playerIndex": 1},
                {"type": 14},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    active_target = _channel(opponent_active_fez, "immediate_disruption_outcome")
    assert active_target["raw"] == [0.0, 0.0]
    assert not active_target["available"]

    missing_active = _evaluate(
        _obs(
            me,
            _player(active=None, bench=[_pokemon(guide.FEZANDIPITI_EX, 100)]),
            [
                {"type": 3, "area": 5, "index": 0, "playerIndex": 1},
                {"type": 14},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    unknown_map = _channel(missing_active, "immediate_disruption_outcome")
    assert unknown_map["raw"] == [0.0, 0.0]
    assert not unknown_map["available"]

    same_yield = _evaluate(
        _obs(
            me,
            _player(active=_pokemon(900, 100), bench=[_pokemon(901, 100)]),
            [
                {"type": 3, "area": 5, "index": 0, "playerIndex": 1},
                {"type": 14},
            ],
            effect_id=guide.BOSS_ORDERS,
        ),
        [[0], [1]],
    )
    no_prize_change = _channel(same_yield, "immediate_disruption_outcome")
    assert no_prize_change["raw"] == [0.0, 0.0]
    assert not no_prize_change["available"]


def test_boss_hammer_and_recovery_prefixes_remain_neutral_before_target_selection() -> None:
    """A source-card prefix is not its later target effect or recovery route."""

    base_me = _player(
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    opponent = _player(active=_pokemon(900, 100, energy=[guide.MIST_ENERGY]))
    for source in (guide.BOSS_ORDERS, guide.ENHANCED_HAMMER, guide.NIGHT_STRETCHER):
        me = dict(base_me)
        me["hand"] = [{"id": source}]
        me["handCount"] = 1
        payload = _evaluate(
            _obs(
                me,
                opponent,
                [
                    {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                    {"type": 14},
                ],
            ),
            [[0], [1]],
        )
        assert _raw(payload, "immediate_disruption_outcome") == [0.0, 0.0]
        assert not _channel(payload, "immediate_disruption_outcome")["available"]
        assert _raw(payload, "unknown_prize_robust_line") == [0.0, 0.0]


def test_battle_cage_does_not_treat_munkidori_ex_as_counter_placement_threat() -> None:
    """Card 139 is a prize effect, not the normal Munkidori counter mover."""

    payload = _evaluate(
        _obs(
            _player(
                hand=[guide.BATTLE_CAGE],
                active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
                bench=[_pokemon(guide.ABRA, 60)],
            ),
            _player(active=_pokemon(139, 100)),
            [
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    cage = _channel(payload, "immediate_disruption_outcome")
    assert cage["raw"] == [0.0, 0.0]
    assert not cage["available"]


def test_fez_ability_is_not_a_second_bench_exposure_decision() -> None:
    """An ability on an already-benched Fez cannot re-score its placement."""

    payload = _evaluate(
        _obs(
            _player(
                active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
                bench=[_pokemon(guide.FEZANDIPITI_EX, 210)],
            ),
            _player(active=_pokemon(900, 100)),
            [
                {"type": 10, "area": 5, "index": 0, "playerIndex": 0},
                {"type": 14},
            ],
            effect_id=guide.FEZANDIPITI_EX,
        ),
        [[0], [1]],
    )
    exposure = _channel(payload, "bench_prize_exposure")
    assert exposure["raw"] == [0.0, 0.0]
    assert not exposure["available"]


def test_terminal_can_end_by_knocking_out_the_last_opponent_pokemon() -> None:
    """No Bench after a proven active KO ends the game even with prizes left."""

    payload = _evaluate(
        _obs(
            _player(
                hand_count=5,
                active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
                deck_count=0,
                prizes=5,
            ),
            _player(active=_pokemon(900, 100), bench=[]),
            [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
        ),
        [[0], [1]],
    )
    terminal = _channel(payload, "terminal_before_forced_draw")
    assert terminal["available"]
    assert terminal["raw"][0] > terminal["raw"][1]
