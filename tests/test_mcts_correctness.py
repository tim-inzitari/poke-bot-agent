from types import SimpleNamespace

import torch
import pytest

from poke_bot import config, features
from poke_bot.agent import PolicyAgent
from poke_bot.batched_infer import LeafPacket
from poke_bot.matchup_adapter_activation import ShadowMatchupAdapterRouter
from poke_bot.matchup_adapters import UNKNOWN_ROUTE, route_for_archetype
from poke_bot.mcts import Child, LeafEvaluator, MCTS, Node


def _state(
    actor: int,
    search_id: int = 0,
    result: int = -1,
    *,
    visible: tuple[int, ...] = (),
):
    obs = SimpleNamespace(
        current=SimpleNamespace(yourIndex=actor, result=result),
        select=SimpleNamespace(option=[object(), object()], minCount=1, maxCount=1),
        visible=visible,
    )
    return SimpleNamespace(observation=obs, searchId=search_id)


def test_opponent_nodes_minimize_root_relative_value() -> None:
    engine = MCTS(
        None,
        [1] * 60,
        puct_c=0.0,
        leaf_backend=lambda packets: packets,
        oracle_mode=True,
    )

    root = Node(state=_state(actor=0))
    high = Node(state=_state(actor=1), parent=root, visit=4, total=3.0)
    low = Node(state=_state(actor=1), parent=root, visit=4, total=-2.0)
    root.children = [
        Child(select=[0], prior=0.5, node=high),
        Child(select=[1], prior=0.5, node=low),
    ]
    assert engine._select_child(root, root_seat=0).node is high

    opponent = Node(state=_state(actor=1))
    high.parent = opponent
    low.parent = opponent
    opponent.children = [
        Child(select=[0], prior=0.5, node=high),
        Child(select=[1], prior=0.5, node=low),
    ]
    assert engine._select_child(opponent, root_seat=0).node is low


def test_leaf_deck_routes_by_simulated_actor() -> None:
    root_deck = [10] * 60
    opponent_deck = [20] * 60
    evaluator = LeafEvaluator(
        None,
        root_deck,
        opponent_deck,
        root_seat=0,
        device=torch.device("cpu"),
        leaf_backend=lambda packets: packets,
    )
    assert evaluator.packet(_state(0).observation).your_deck == root_deck
    assert evaluator.packet(_state(1).observation).your_deck == opponent_deck


def test_search_branch_shadow_routers_fork_without_reaching_model(monkeypatch) -> None:
    import poke_bot.matchup_adapter_activation as activation

    monkeypatch.setattr(
        activation,
        "visible_opponent_card_ids",
        lambda obs: frozenset(getattr(obs, "visible", ())),
    )
    shadow = ShadowMatchupAdapterRouter()
    root_obs = _state(0, visible=(646,)).observation
    assert shadow.observe(root_obs).route == UNKNOWN_ROUTE
    root = Node(
        state=_state(0),
        matchup_shadow_router=shadow.fork(),
    )
    engine = MCTS(
        None,
        [1] * 60,
        leaf_backend=lambda packets: packets,
        oracle_mode=True,
        matchup_shadow_router=shadow.fork(),
    )
    first = engine._shadow_router_for_state(
        root,
        _state(0, visible=(646,)).observation,
        root_seat=0,
    )
    second = engine._shadow_router_for_state(
        root,
        _state(0, visible=(646,)).observation,
        root_seat=0,
    )
    assert first is not None and second is not None
    expected = route_for_archetype("marnie-s-grimmsnarl-ex")
    assert first.game_router.recognizer.last_decision.route == expected
    assert second.game_router.recognizer.last_decision.route == expected
    assert first.model_route == second.model_route == UNKNOWN_ROUTE
    # An opponent-actor branch is deliberately not observed through the root
    # agent's recognizer because `yourIndex` would invert the matchup.
    opponent = engine._shadow_router_for_state(
        root,
        _state(1, visible=(646,)).observation,
        root_seat=0,
    )
    assert opponent is not None
    assert opponent.game_router.recognizer.last_decision.route == UNKNOWN_ROUTE


def test_single_tree_search_evaluates_before_next_selection(monkeypatch) -> None:
    from poke_bot import cg_env

    events: list[str] = []
    root_state = _state(actor=0, search_id=0)

    monkeypatch.setattr(config.SEARCH, "leaf_batch_mcts", False)
    monkeypatch.setattr(config.SEARCH, "min_sim_completion_ratio", 0.5)
    monkeypatch.setattr(cg_env, "to_observation", lambda obs: obs)
    monkeypatch.setattr(cg_env, "build_search_inputs", lambda *a, **k: {})
    monkeypatch.setattr(cg_env, "search_begin", lambda *a, **k: root_state)
    monkeypatch.setattr(cg_env, "search_end", lambda: events.append("end"))
    monkeypatch.setattr(features, "assert_info_set", lambda obs: None)
    monkeypatch.setattr(features, "enumerate_action_combos", lambda obs: [[0], [1]])

    def _step(_search_id, select):
        events.append(f"step{select[0]}")
        return _state(actor=1, search_id=10 + select[0])

    monkeypatch.setattr(cg_env, "search_step", _step)

    eval_count = 0

    def _backend(packets):
        nonlocal eval_count
        assert len(packets) == 1  # no blind within-tree wave
        events.append(f"eval{eval_count}")
        packet = packets[0]
        if eval_count == 0:
            out = LeafPacket(
                packet.obs,
                packet.your_deck,
                packet.root_seat,
                value=0.0,
                priors=[0.9, 0.1],
                combos=[[0], [1]],
            )
        else:
            out = LeafPacket(
                packet.obs,
                packet.your_deck,
                packet.root_seat,
                value=-1.0 if eval_count == 1 else 1.0,
                priors=[],
                combos=[],
            )
        eval_count += 1
        return [out]

    engine = MCTS(
        None,
        [10] * 60,
        opponent_deck_guess=[20] * 60,
        puct_c=0.0,
        leaf_backend=_backend,
        oracle_mode=True,
    )
    result = engine.search(
        root_state.observation,
        max_sims=2,
        move_time_s=10.0,
    )
    assert result.sims_run == 2
    assert events[:5] == ["eval0", "step0", "eval1", "step1", "eval2"]
    assert result.target.diagnostics["max_depth"] == 1
    assert result.target.diagnostics["unique_nodes"] == 3
    assert result.target.diagnostics["unique_expanded_nodes"] == 1
    assert result.target.diagnostics["leaf_evaluations"] == 3
    assert result.target.diagnostics["mean_depth"] == pytest.approx(2 / 3)
    assert result.target.diagnostics["mean_branching"] == 2
    assert result.target.diagnostics["sims_per_s"] > 0
    assert result.target.diagnostics["trusted"] is False
    assert result.target.diagnostics["belief_mode"] == "single_determinization"


def test_action_enumeration_is_complete_ordered_or_fails() -> None:
    obs = SimpleNamespace(
        select=SimpleNamespace(
            option=[object(), object(), object()],
            minCount=1,
            maxCount=2,
        )
    )
    full = features.enumerate_action_combos(obs, max_combos=20)
    assert {len(combo) for combo in full} == {1, 2}
    assert full.total_count == 9
    assert full.truncated is False
    assert [0, 1] in full and [1, 0] in full

    try:
        features.enumerate_action_combos(obs, max_combos=3)
    except features.ActionSpaceTooLarge as exc:
        assert "9 actions" in str(exc)
    else:
        raise AssertionError("trusted action enumeration must not truncate")


def test_factorized_actions_cover_ordered_space_without_materializing_it() -> None:
    obs = SimpleNamespace(
        select=SimpleNamespace(
            option=[object(), object(), object()],
            minCount=1,
            maxCount=2,
        )
    )

    generated: set[tuple[int, ...]] = set()

    def walk(prefix: list[int]) -> None:
        for candidate in features.factorized_action_candidates(obs, prefix):
            if candidate == prefix:
                generated.add(tuple(prefix))
            else:
                walk(candidate)

    walk([])
    exhaustive = {
        tuple(combo) for combo in features.enumerate_action_combos(obs, max_combos=20)
    }
    assert generated == exhaustive

    huge = SimpleNamespace(
        select=SimpleNamespace(
            option=[object()] * 12,
            minCount=9,
            maxCount=9,
        )
    )
    action = list(range(9))
    stages = features.factorized_teacher_forcing_stages(huge, action)
    assert len(stages) == 9
    assert max(len(candidates) for candidates, _ in stages) == 12
    assert all(candidates[target] == action[: i + 1] for i, (candidates, target) in enumerate(stages))
    with pytest.raises(features.ActionSpaceTooLarge, match="79833600 actions"):
        features.enumerate_action_combos(huge)


def test_privileged_deck_and_single_world_search_require_oracle_mode() -> None:
    with pytest.raises(ValueError, match="oracle-only"):
        PolicyAgent(model=None, deck=[1] * 60, use_mcts=True)
    with pytest.raises(ValueError, match="privileged"):
        PolicyAgent(
            model=None,
            deck=[1] * 60,
            opponent_deck=[2] * 60,
            use_mcts=False,
        )
