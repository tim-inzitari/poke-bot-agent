"""Focused r219 tests for capability-gated finite chance in belief MCTS."""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BELIEF_MCTS_PATH = ROOT / "poke_bot/belief_mcts.py"


@dataclass
class _MCTSResult:
    select: list[int]
    target: object
    sims_run: int
    elapsed_s: float


def _load_belief_mcts_without_torch(monkeypatch: pytest.MonkeyPatch):
    """Import the real module with just its import-time runtime faked."""

    if "poke_bot" not in sys.modules:
        package = types.ModuleType("poke_bot")
        package.__path__ = [str(ROOT / "poke_bot")]
        monkeypatch.setitem(sys.modules, "poke_bot", package)

    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda _name: "cpu"
    fake_torch.tensor = lambda value: value

    fake_config = types.ModuleType("poke_bot.config")
    fake_config.SEARCH = SimpleNamespace(puct_c=1.0)
    fake_config.MODEL = SimpleNamespace(max_context=8)

    fake_features = types.ModuleType("poke_bot.features")
    fake_features.SparseVector = object
    fake_features.assert_info_set = lambda _obs: None
    fake_features.ordered_action_count = lambda _obs: 1
    fake_features.build_option_tokens = lambda _obs, _actions: None
    fake_features.enumerate_action_combos = lambda _obs: []
    fake_features.factorized_action_candidates = lambda _obs, _prefix: []
    fake_features.build_board_tokens = lambda _obs, _deck: None

    fake_cg_env = types.ModuleType("poke_bot.cg_env")
    fake_cg_env.to_observation = lambda _observation: None
    fake_cg_env.search_begin = lambda *_args, **_kwargs: None
    fake_cg_env.search_step = lambda *_args, **_kwargs: None
    fake_cg_env.search_end = lambda: None

    fake_batched_infer = types.ModuleType("poke_bot.batched_infer")
    fake_batched_infer.LeafPacket = object
    fake_batched_infer.forward_leaf_batch = lambda _packets: []

    fake_belief = types.ModuleType("poke_bot.belief")
    fake_belief.EmpiricalDeckPosterior = object
    fake_belief.HiddenStateParticle = object
    fake_belief.NeuralBeliefPriors = object
    fake_belief.PublicBeliefHistory = object
    fake_belief.assert_deployment_observation = lambda _obs: None
    fake_belief.simulator_version = lambda: "r219-test-simulator"

    fake_blackwell = types.ModuleType("poke_bot.blackwell_heads")
    fake_blackwell.blackwell_strategy_heads_enabled = lambda: False
    fake_blackwell.root_value_bias_from_lethal = lambda _value: 0.0

    fake_adapter = types.ModuleType("poke_bot.matchup_adapter_activation")
    fake_adapter.ShadowMatchupAdapterRouter = object

    fake_mcts = types.ModuleType("poke_bot.mcts")
    fake_mcts.GameClock = object
    fake_mcts.MCTSResult = _MCTSResult

    fake_model = types.ModuleType("poke_bot.model")
    fake_model.TemporalCabtTransformer = object
    fake_model.card_prior_logits_or_uniform = lambda *_args, **_kwargs: None

    fake_replay = types.ModuleType("poke_bot.replay_import")
    fake_replay.assert_info_set = lambda _obs: None

    fake_targets = types.ModuleType("poke_bot.search_targets")
    fake_targets.build_search_target = lambda *_args, **_kwargs: None
    fake_targets.select_by_visits = lambda combos, _visits: list(combos[0])

    for name, module in {
        "torch": fake_torch,
        "poke_bot.batched_infer": fake_batched_infer,
        "poke_bot.belief": fake_belief,
        "poke_bot.blackwell_heads": fake_blackwell,
        "poke_bot.cg_env": fake_cg_env,
        "poke_bot.config": fake_config,
        "poke_bot.features": fake_features,
        "poke_bot.matchup_adapter_activation": fake_adapter,
        "poke_bot.mcts": fake_mcts,
        "poke_bot.model": fake_model,
        "poke_bot.replay_import": fake_replay,
        "poke_bot.search_targets": fake_targets,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "poke_bot._r219_exact_finite_chance_test"
    spec = importlib.util.spec_from_file_location(module_name, BELIEF_MCTS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _observation(*, context: int, result: int = -1) -> dict:
    return {
        "logs": [],
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}, {"type": 13}],
        },
        "current": {
            "turn": 1,
            "yourIndex": 0,
            "result": result,
            "players": [],
        },
    }


class _ExactCapability:
    def __init__(self, module, outcomes: tuple[object, ...]):
        self.module = module
        self.outcomes = outcomes
        self.calls: list[object] = []

    def enumerate_exact_finite_chance(self, search_state):
        self.calls.append(search_state)
        return self.module.ExactFiniteChanceExpansion(
            outcomes=self.outcomes,
            force_enumeration_receipt="force-enumeration-receipt",
            probability_receipt="exact-probability-receipt",
        )


def _outcomes(
    module,
    results: list[int],
    probability: Fraction,
) -> tuple[object, ...]:
    return tuple(
        module.ExactFiniteChanceOutcome(
            label=f"outcome-{index}",
            probability=probability,
            successor=SimpleNamespace(
                searchId=100 + index,
                observation=_observation(context=0, result=result),
            ),
            successor_receipt=f"successor-{index}",
            future_legality_receipt=f"terminal-or-legal-{index}",
        )
        for index, result in enumerate(results)
    )


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    capability: object | None,
):
    """Build a one-action private search without importing the live cg API."""

    root_observation = _observation(context=0)
    chance_observation = _observation(context=46)
    root_state = SimpleNamespace(searchId=1, observation=root_observation)
    chance_state = SimpleNamespace(searchId=2, observation=chance_observation)
    calls: list[tuple[int, tuple[int, ...]]] = []

    particle = SimpleNamespace(
        search_inputs={"private": "r219"},
        opponent_deck_digest="r219-deck",
        support_mode="r219-test",
        support_repairs=0,
    )

    monkeypatch.setattr(
        module,
        "assert_deployment_observation",
        lambda _observation: None,
    )
    monkeypatch.setattr(module.features, "assert_info_set", lambda _obs: None)
    monkeypatch.setattr(
        module,
        "assert_public_info_set",
        lambda _observation: None,
    )
    monkeypatch.setattr(
        module.cg_env,
        "to_observation",
        lambda _observation: SimpleNamespace(
            current=SimpleNamespace(yourIndex=0)
        ),
    )
    monkeypatch.setattr(
        module.cg_env,
        "search_begin",
        lambda *_args, **_kwargs: root_state,
    )

    def search_step(search_id: int, action: list[int]):
        calls.append((search_id, tuple(action)))
        if search_id == root_state.searchId:
            return chance_state
        assert search_id == chance_state.searchId
        # The sampled fallback still has a complete fake simulator response.
        return SimpleNamespace(
            searchId=20 + action[0],
            observation=_observation(context=0, result=0 if action[0] == 0 else 1),
        )

    monkeypatch.setattr(module.cg_env, "search_step", search_step)
    monkeypatch.setattr(module.cg_env, "search_end", lambda: None)
    monkeypatch.setattr(
        module.features,
        "enumerate_action_combos",
        lambda obs: [[index] for index, _option in enumerate(obs["select"]["option"])],
    )
    monkeypatch.setattr(
        module.features,
        "build_option_tokens",
        lambda _obs, _actions: None,
    )
    monkeypatch.setattr(
        module,
        "build_search_target",
        lambda combos, visits, value, **kwargs: SimpleNamespace(
            action_combos=[list(combo) for combo in combos],
            visits=list(visits),
            value=float(value),
            diagnostics=dict(kwargs["diagnostics"]),
        ),
    )
    monkeypatch.setattr(
        module,
        "select_by_visits",
        lambda combos, visits: list(
            combos[max(range(len(combos)), key=lambda index: (visits[index], -index))]
        ),
    )

    engine = object.__new__(module.BeliefMCTS)
    engine.min_trusted_sims = 1
    engine.particle_count = 1
    engine.max_depth = 8
    engine.max_context = 8
    engine.puct_c = 1.0
    engine.rng = random.Random(7)
    engine.own_deck = tuple([1] * 60)
    engine.posterior = SimpleNamespace(
        sample_particle=lambda *_args, **_kwargs: particle,
    )
    engine.matchup_shadow_router = None
    engine.leaf_evaluator_source = "r219-fake"
    engine.leaf_evaluator_checkpoint_digest = None
    engine.exact_finite_chance_capability = capability
    engine.finite_chance_capability = capability
    engine.convergence_min_sims = 32
    engine.convergence_stable_sims = 16
    engine.convergence_visit_share = 0.9
    engine.convergence_q_margin = 0.05
    engine._telemetry_mark = lambda: None
    engine._telemetry_since = lambda _marker: {}
    engine._root_neural_priors = lambda **_kwargs: SimpleNamespace(
        lethal_threat_logit=None,
        prize_race=None,
    )

    def evaluate(node, _obs, **_kwargs):
        node.network_evaluated = True
        node.factorized = False
        node.total_action_count = 1
        node.bootstrap_value = 0.0
        node.edges = [module.BeliefEdge([0], 1.0)]
        return 0.0, True

    engine._evaluate_node_cached = evaluate
    return engine, root_observation, chance_state, calls


def _run_one_action(engine, root_observation):
    return engine._search_impl(
        root_observation,
        belief_history=SimpleNamespace(observe=lambda _obs: None),
        root_history_boards=(),
        root_history_previous_actions=(),
        max_sims=1,
        move_time_s=1.0,
    )


def test_capability_enumerates_a_half_coin_and_backs_up_exact_expectation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_belief_mcts_without_torch(monkeypatch)
    capability = _ExactCapability(
        module,
        _outcomes(module, [0, 1], Fraction(1, 2)),
    )
    engine, root_observation, chance_state, calls = _engine(
        monkeypatch,
        module,
        capability=capability,
    )

    result = _run_one_action(engine, root_observation)
    telemetry = result.target.diagnostics

    assert capability.calls == [chance_state]
    assert calls == [(1, (0,))]
    assert result.sims_run == 1
    assert result.target.value == pytest.approx(0.0)
    assert telemetry["finite_chance_enumerations"] == 1
    assert telemetry["finite_chance_outcomes_enumerated"] == 2
    assert telemetry["finite_chance_weighted_backup_count"] == 1
    assert telemetry["finite_chance_forced_successor_transitions"] == 2
    assert telemetry["exact_terminal_results_seen"] == 2
    assert telemetry["chance_samples"] == 0
    assert telemetry["sampled_unforceable_chance_nodes"] == 0
    assert telemetry["r207_exact_finite_chance_probability_weighted_expectation_claimed"] is False


def test_capability_enumerates_a_sixth_die_and_uses_all_six_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_belief_mcts_without_torch(monkeypatch)
    # Four wins and two losses give (+4 - 2) / 6 = 1/3.  This catches an
    # implementation that uses a coin-only path or drops low-prior outcomes.
    capability = _ExactCapability(
        module,
        _outcomes(module, [0, 0, 0, 0, 1, 1], Fraction(1, 6)),
    )
    engine, root_observation, _chance_state, calls = _engine(
        monkeypatch,
        module,
        capability=capability,
    )

    result = _run_one_action(engine, root_observation)
    telemetry = result.target.diagnostics

    assert calls == [(1, (0,))]
    assert result.target.value == pytest.approx(1.0 / 3.0)
    assert telemetry["finite_chance_enumerations"] == 1
    assert telemetry["finite_chance_outcomes_enumerated"] == 6
    assert telemetry["finite_chance_weighted_backup_count"] == 1
    assert telemetry["finite_chance_forced_successor_transitions"] == 6
    assert telemetry["exact_terminal_results_seen"] == 6
    assert telemetry["simulator_transitions"] == 7
    assert telemetry["chance_samples"] == 0


def test_missing_capability_stops_at_pre_random_leaf_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_belief_mcts_without_torch(monkeypatch)
    engine, root_observation, _chance_state, calls = _engine(
        monkeypatch,
        module,
        capability=None,
    )

    result = _run_one_action(engine, root_observation)
    telemetry = result.target.diagnostics

    assert len(calls) == 1  # root decision only; no guessed chance dispatch
    assert telemetry["finite_chance_enumerations"] == 0
    assert telemetry["finite_chance_outcomes_enumerated"] == 0
    assert telemetry["finite_chance_weighted_backup_count"] == 0
    assert telemetry["chance_samples"] == 0
    assert telemetry["sampled_unforceable_chance_nodes"] == 0
    assert telemetry["sampled_unforceable_chance_reasons"] == {}
    assert telemetry["unforceable_chance_boundary_nodes"] == 1
    assert telemetry["unforceable_chance_boundary_leaf_evaluations"] == 1
    assert telemetry["unforceable_chance_boundary_reasons"] == {
        "no_exact_finite_chance_capability": 1
    }
    assert telemetry["chance_mode"] == "unforceable_random_pre_boundary_leaf"
    assert telemetry["private_unforceable_chance_samples_prohibited"] is True
    assert telemetry["seed_hunting_or_pre_randomization_prohibited"] is True


def test_exact_chance_revisits_each_child_past_its_legal_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-chance line becomes best only after both child decisions run."""

    module = _load_belief_mcts_without_torch(monkeypatch)
    heads = _observation(context=0)
    heads["child"] = "heads"
    tails = _observation(context=0)
    tails["child"] = "tails"
    outcomes = (
        module.ExactFiniteChanceOutcome(
            label="heads",
            probability=Fraction(1, 2),
            successor=SimpleNamespace(searchId=100, observation=heads),
            successor_receipt="heads-child",
            future_legality_receipt="heads-legal",
        ),
        module.ExactFiniteChanceOutcome(
            label="tails",
            probability=Fraction(1, 2),
            successor=SimpleNamespace(searchId=101, observation=tails),
            successor_receipt="tails-child",
            future_legality_receipt="tails-legal",
        ),
    )
    capability = _ExactCapability(module, outcomes)
    expansion = capability.enumerate_exact_finite_chance(object())
    calls: list[tuple[int, tuple[int, ...]]] = []

    def search_step(search_id: int, action: list[int]):
        calls.append((search_id, tuple(action)))
        if search_id == 100:
            assert action == [0]  # Heads prefers action zero.
        elif search_id == 101:
            assert action == [1]  # Tails prefers action one.
        else:  # pragma: no cover - makes an accidental root advance obvious.
            raise AssertionError(f"unexpected child search id {search_id}")
        return SimpleNamespace(
            searchId=200 + search_id,
            observation=_observation(context=0, result=0),
        )

    monkeypatch.setattr(module.cg_env, "search_step", search_step)
    monkeypatch.setattr(module.features, "assert_info_set", lambda _obs: None)
    monkeypatch.setattr(module, "assert_public_info_set", lambda _obs: None)
    monkeypatch.setattr(
        module.features,
        "build_option_tokens",
        lambda _obs, _actions: None,
    )
    monkeypatch.setattr(
        module.features,
        "build_board_tokens",
        lambda _obs, _deck: None,
    )

    engine = object.__new__(module.BeliefMCTS)
    engine.puct_c = 1.0
    engine.max_depth = 8
    engine.max_context = 8
    engine.own_deck = tuple([1] * 60)
    engine.exact_finite_chance_capability = capability
    engine.finite_chance_capability = capability
    expansion = engine._validate_exact_finite_chance_expansion(expansion)

    def evaluate(node, observation, **_kwargs):
        node.network_evaluated = True
        node.factorized = False
        node.total_action_count = 2
        node.bootstrap_value = 0.0
        if observation["child"] == "heads":
            node.edges = [module.BeliefEdge([0], 1.0), module.BeliefEdge([1], 0.0)]
        else:
            node.edges = [module.BeliefEdge([0], 0.0), module.BeliefEdge([1], 1.0)]
        return 0.0, True

    engine._evaluate_node_cached = evaluate
    parent = module.BeliefNode("pre-chance", actor=0, depth=0)
    parent_edge = module.BeliefEdge([0], 1.0)
    parent.edges = [parent_edge]
    branch = module._BranchHistory(
        boards={0: [], 1: []},
        previous_actions={0: [], 1: []},
        last_action={0: None, 1: None},
    )
    particle = SimpleNamespace()
    cache = module._DecisionEvaluationCache()

    first = engine._evaluate_exact_finite_chance(
        parent=parent,
        edge=parent_edge,
        expansion=expansion,
        root_seat=0,
        particle=particle,
        branch=branch,
        depth=1,
        evaluation_cache=cache,
        scenario_prefix="r219-test",
    )
    module.BeliefMCTS._backup([parent], [parent_edge], first.value)
    # Both forced children have an initial policy/value leaf, but neither has
    # yet dispatched its next legal controlled action.
    assert first.value == pytest.approx(0.0)
    assert first.leaf_evaluations == 2
    assert calls == []

    second = engine._evaluate_exact_finite_chance(
        parent=parent,
        edge=parent_edge,
        expansion=expansion,
        root_seat=0,
        particle=particle,
        branch=branch,
        depth=1,
        evaluation_cache=cache,
        scenario_prefix="r219-test",
    )
    module.BeliefMCTS._backup([parent], [parent_edge], second.value)

    # An alternative pre-chance line valued at +0.25 loses only after the
    # two post-coin decisions are searched.  A one-ply child leaf scorer would
    # remain at zero and cannot establish this preference.
    assert second.value == pytest.approx(1.0)
    assert parent_edge.q() == pytest.approx(0.5)
    assert first.value < 0.25 < second.value
    assert calls == [(100, (0,)), (101, (1,))]
    assert second.simulator_transitions == 4
    assert second.max_simulator_steps == 2


@pytest.mark.parametrize("minimum", (0, -1))
def test_minimum_trusted_simulations_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
    minimum: int,
) -> None:
    module = _load_belief_mcts_without_torch(monkeypatch)

    with pytest.raises(ValueError, match="at least one completed backup"):
        module.BeliefMCTS(
            model=None,
            own_deck=[1] * 60,
            posterior=object(),
            checkpoint_digest="sha256:" + "0" * 64,
            model_generation=0,
            leaf_backend=lambda _packets: [],
            min_trusted_sims=minimum,
        )
