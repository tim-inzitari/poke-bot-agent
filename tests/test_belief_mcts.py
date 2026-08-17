from collections import Counter
from types import SimpleNamespace

import pytest

from poke_bot import features
from poke_bot.belief import (
    BeliefSupportError,
    EmpiricalDeckPosterior,
    PublicBeliefHistory,
    _basic_pokemon_ids,
    simulator_version,
)
from poke_bot.belief_mcts import (
    BeliefEdge,
    BeliefMCTS,
    BeliefNode,
    _BranchHistory,
    _DecisionEvaluationCache,
    factorize_visit_policy,
    information_state_fingerprint,
    is_explicit_chance,
)


def _card(card_id: int, serial: int, player: int) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


def _observation(*, leaked_hand=False, search_token="opaque-a") -> dict:
    own_hand = [_card(1, serial, 0) for serial in range(1, 6)]
    own_active = _card(1, 6, 0)
    opp_active = _card(20, 100, 1)
    opp_discard = _card(10, 101, 1)
    return {
        "search_begin_input": search_token,
        "logs": [],
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}, {"type": 13}],
            "deck": None,
        },
        "current": {
            "turn": 3,
            "yourIndex": 0,
            "result": -1,
            "stadium": [],
            "players": [
                {
                    "active": [own_active],
                    "bench": [],
                    "deckCount": 48,
                    "discard": [],
                    "prize": [None] * 6,
                    "handCount": 5,
                    "hand": own_hand,
                },
                {
                    "active": [opp_active],
                    "bench": [],
                    "deckCount": 47,
                    "discard": [opp_discard],
                    "prize": [None] * 6,
                    "handCount": 5,
                    "hand": [_card(10, 102, 1)] if leaked_hand else None,
                },
            ],
        },
    }


def test_public_history_rejects_opponent_private_hand() -> None:
    with pytest.raises(ValueError, match="hidden-state leakage"):
        PublicBeliefHistory().observe(_observation(leaked_hand=True))
    leaked = _observation()
    leaked["current"]["players"][1]["deckOrder"] = [10] * 47
    with pytest.raises(ValueError, match="privileged fields"):
        PublicBeliefHistory().observe(leaked)


def test_simulator_version_uses_packaged_cg_runtime(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "agent"
    cg = runtime / "cg"
    cg.mkdir(parents=True)
    (cg / "api.py").write_bytes(b"packaged api")
    (cg / "sim.py").write_bytes(b"packaged sim")
    (cg / "libcg.so").write_bytes(b"packaged native")
    monkeypatch.setenv("CG_LIB_PATH", str(runtime))

    digest = simulator_version()

    assert digest.startswith("competition-libcg-sha256:")


def test_empirical_posterior_conditions_on_public_history_and_conserves_cards() -> None:
    supporting = [1] * 20 + [10] * 20 + [20] * 20
    unsupported = [1] * 30 + [10] * 30
    posterior = EmpiricalDeckPosterior([supporting, unsupported])
    history = PublicBeliefHistory()
    obs = _observation()
    history.observe(obs)
    rows = posterior.posterior(history)
    assert [row[0].cards for row in rows] == [tuple(sorted(supporting))]
    assert posterior.config["uses_baseline_identity"] is False

    import random

    particle = posterior.sample_particle(
        obs,
        own_deck=[1] * 60,
        history=history,
        rng=random.Random(4),
    )
    predicted = Counter()
    for key in ("opponent_deck", "opponent_prize", "opponent_hand", "opponent_active"):
        predicted.update(particle.search_inputs[key])
    predicted.update([20, 10])  # public active + discard
    assert predicted == Counter(supporting)
    own_predicted = Counter(particle.search_inputs["your_deck"])
    own_predicted.update(particle.search_inputs["your_prize"])
    own_predicted.update([1] * 6)  # own hand + active
    assert own_predicted == Counter([1] * 60)
    assert particle.public_history_digest == history.digest


def test_particle_conservation_accounts_for_visible_looking_cards() -> None:
    obs = _observation()
    looking = _card(1, 7, 0)
    obs["current"]["looking"] = [looking]
    obs["select"]["deck"] = [looking]
    obs["current"]["players"][0]["deckCount"] = 47
    posterior = EmpiricalDeckPosterior([[1] * 20 + [10] * 20 + [20] * 20])
    history = PublicBeliefHistory()
    history.observe(obs)
    import random

    particle = posterior.sample_particle(
        obs,
        own_deck=[1] * 60,
        history=history,
        rng=random.Random(5),
    )
    # libcg keeps the visible serialized deck and ignores your_deck here.
    assert particle.search_inputs["your_deck"] == []
    assert len(particle.search_inputs["your_prize"]) == 6


def test_self_private_history_conditions_facedown_setup_active() -> None:
    before = _observation()
    before["current"]["players"][0]["active"] = []
    before["current"]["players"][0]["deckCount"] = 47
    before["current"]["players"][0]["hand"].extend(
        [_card(1, 7, 0), _card(1, 8, 0)]
    )
    before["current"]["players"][0]["handCount"] = 7
    before["select"]["option"] = [{"type": 3, "area": 2, "index": 0}]
    history = PublicBeliefHistory()
    history.observe(before)
    history.record_action(before, [0])
    assert history.self_facedown_active_card == 1

    after = _observation()
    after["current"]["players"][0]["active"] = [None]
    after["current"]["players"][0]["deckCount"] = 47
    after["current"]["players"][0]["hand"].append(_card(1, 7, 0))
    after["current"]["players"][0]["handCount"] = 6
    posterior = EmpiricalDeckPosterior([[1] * 20 + [10] * 20 + [20] * 20])
    import random

    particle = posterior.sample_particle(
        after,
        own_deck=[1] * 60,
        history=history,
        rng=random.Random(6),
    )
    assert len(particle.search_inputs["your_deck"]) == 47


def test_particle_reserves_basic_before_sampling_other_hidden_zones() -> None:
    import random

    basic = next(iter(_basic_pokemon_ids()))
    filler = next(card_id for card_id in range(1, 1000) if card_id not in _basic_pokemon_ids())
    deck = [basic] + [filler] * 59
    obs = _observation()
    opponent = obs["current"]["players"][1]
    opponent["active"] = [None]
    opponent["discard"] = []
    opponent["deckCount"] = 48
    history = PublicBeliefHistory()
    history.observe(obs)
    posterior = EmpiricalDeckPosterior([deck])
    for seed in range(64):
        particle = posterior.sample_particle(
            obs,
            own_deck=[1] * 60,
            history=history,
            rng=random.Random(seed),
        )
        assert particle.search_inputs["opponent_active"] == [basic]
        assigned = (
            particle.search_inputs["opponent_active"]
            + particle.search_inputs["opponent_hand"]
            + particle.search_inputs["opponent_prize"]
            + particle.search_inputs["opponent_deck"]
        )
        assert Counter(assigned) == Counter(deck)


def test_unsupported_public_card_uses_history_consistent_repaired_prior() -> None:
    import random

    obs = _observation()
    obs["current"]["players"][1]["active"] = [_card(30, 100, 1)]
    posterior = EmpiricalDeckPosterior(
        [[1] * 20 + [10] * 20 + [20] * 20]
    )
    history = PublicBeliefHistory()
    history.observe(obs)
    particle = posterior.sample_particle(
        obs,
        own_deck=[1] * 60,
        history=history,
        rng=random.Random(9),
    )
    assert particle.support_mode == "observable_history_conditioned_repair"
    assert Counter(particle.opponent_deck)[30] >= 1
    predicted = Counter(
        particle.search_inputs["opponent_deck"]
        + particle.search_inputs["opponent_prize"]
        + particle.search_inputs["opponent_hand"]
        + [30, 10]
    )
    assert predicted == Counter(particle.opponent_deck)


def test_truly_empty_setup_support_is_clean_belief_failure() -> None:
    import random

    filler = next(
        card_id
        for card_id in range(1, 1000)
        if card_id not in _basic_pokemon_ids()
    )
    obs = _observation()
    opponent = obs["current"]["players"][1]
    opponent["active"] = [None]
    opponent["discard"] = []
    opponent["deckCount"] = 48
    history = PublicBeliefHistory()
    history.observe(obs)
    posterior = EmpiricalDeckPosterior([[filler] * 60])
    with pytest.raises(
        BeliefSupportError,
        match="observable-history-consistent particle support is empty",
    ):
        posterior.sample_particle(
            obs,
            own_deck=[1] * 60,
            history=history,
            rng=random.Random(10),
        )


@pytest.mark.parametrize(
    ("n_options", "min_count", "max_count", "expected"),
    [
        (10, 0, 4, 5_861),
        (9, 6, 6, 60_480),
        (12, 9, 9, 79_833_600),
    ],
)
def test_large_ordered_action_space_uses_exact_hierarchical_support(
    n_options: int,
    min_count: int,
    max_count: int,
    expected: int,
) -> None:
    obs = _observation()
    obs["select"]["option"] = [{"type": 14} for _ in range(n_options)]
    obs["select"]["minCount"] = min_count
    obs["select"]["maxCount"] = max_count
    assert features.ordered_action_count(obs) == expected
    engine = object.__new__(BeliefMCTS)
    engine.own_deck = tuple([1] * 60)

    def leaf_eval(packets):
        packet = packets[0]
        combos = list(packet.action_combos_override)
        return [
            SimpleNamespace(
                combos=combos,
                priors=[1.0 / len(combos)] * len(combos),
                value=0.0,
            )
        ]

    engine.leaf_eval = leaf_eval
    root = BeliefNode(
        fingerprint=information_state_fingerprint(obs),
        actor=0,
        depth=0,
    )
    branch = _BranchHistory(
        boards={0: [], 1: []},
        previous_actions={0: [], 1: []},
        last_action={0: None, 1: None},
    )
    engine._evaluate_node(
        root,
        obs,
        root_seat=0,
        particle=None,
        branch=branch,
    )
    assert root.factorized is True
    expected_root = [[i] for i in range(n_options)]
    if min_count == 0:
        expected_root.append([])
    assert [edge.action for edge in root.edges] == expected_root
    assert root.total_action_count == expected
    prefix = list(range(max_count))
    assert engine._factorized_action_complete(obs, prefix[:-1], prefix)


def test_information_state_fingerprint_ignores_opaque_search_token() -> None:
    first = _observation(search_token="opaque-hidden-state-a")
    second = _observation(search_token="opaque-hidden-state-b")
    assert information_state_fingerprint(first) == information_state_fingerprint(second)


def test_explicit_coin_chance_detection() -> None:
    obs = _observation()
    assert is_explicit_chance(obs) is False
    obs["select"]["context"] = 46
    assert is_explicit_chance(obs) is True


def test_belief_tree_minimax_uses_root_relative_values() -> None:
    engine = object.__new__(BeliefMCTS)
    engine.puct_c = 0.0
    high = BeliefEdge([0], 0.5, visit=2, total=2.0)
    low = BeliefEdge([1], 0.5, visit=2, total=-1.0)
    root_node = BeliefNode("root", actor=0, depth=0, edges=[high, low], visit=4)
    opp_node = BeliefNode("opp", actor=1, depth=1, edges=[high, low], visit=4)
    assert engine._select_edge(root_node, root_seat=0) is high
    assert engine._select_edge(opp_node, root_seat=0) is low


def test_leaf_evaluation_cache_reuses_only_the_same_attested_scenario() -> None:
    """The cache copies frozen output but keeps each node's visits separate."""

    engine = object.__new__(BeliefMCTS)
    engine.own_deck = tuple([1] * 60)
    engine.matchup_model_route = -1
    calls: list[list[list[int]]] = []

    def leaf_eval(packets):
        packet = packets[0]
        combos = [list(combo) for combo in packet.action_combos_override]
        calls.append(combos)
        return [
            SimpleNamespace(
                combos=combos,
                priors=[1.0 / len(combos)] * len(combos),
                value=0.25,
            )
        ]

    engine.leaf_eval = leaf_eval
    obs = _observation()
    branch = _BranchHistory(
        boards={0: [], 1: []},
        previous_actions={0: [], 1: []},
        last_action={0: None, 1: None},
    )
    cache = _DecisionEvaluationCache()
    first = BeliefNode("first", actor=0, depth=1)
    value, issued = engine._evaluate_node_cached(
        first,
        obs,
        root_seat=0,
        particle=None,
        branch=branch,
        evaluation_cache=cache,
        hidden_or_chance_scenario="native-attested-same-scenario",
    )
    assert issued is True
    assert value == pytest.approx(0.25)

    second = BeliefNode("second", actor=0, depth=1)
    cached_value, cached_issued = engine._evaluate_node_cached(
        second,
        obs,
        root_seat=0,
        particle=None,
        branch=branch,
        evaluation_cache=cache,
        hidden_or_chance_scenario="native-attested-same-scenario",
    )
    assert cached_issued is False
    assert cached_value == pytest.approx(0.25)
    assert len(calls) == 1
    assert cache.hits == 1
    assert cache.misses == 1
    assert first.edges is not second.edges
    assert first.edges[0].visit == second.edges[0].visit == 0

    third = BeliefNode("third", actor=0, depth=1)
    _value, changed_scenario_issued = engine._evaluate_node_cached(
        third,
        obs,
        root_seat=0,
        particle=None,
        branch=branch,
        evaluation_cache=cache,
        hidden_or_chance_scenario="different-hidden-or-chance-scenario",
    )
    assert changed_scenario_issued is True
    assert len(calls) == 2


def test_visit_policy_factorization_preserves_ordered_mass() -> None:
    obs = SimpleNamespace(
        select=SimpleNamespace(
            option=[object(), object()], minCount=1, maxCount=1
        )
    )
    combos = [[0], [1]]
    rows = factorize_visit_policy(obs, combos, [0.25, 0.75], [1])
    assert len(rows) == 1
    assert rows[0]["action_combos"] == combos
    assert rows[0]["policy"] == pytest.approx([0.25, 0.75])
    assert rows[0]["selected_index"] == 1
