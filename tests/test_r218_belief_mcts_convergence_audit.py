from __future__ import annotations

import importlib.util
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from poke_bot.r215_full_turn_belief_mcts import (
    R215FullTurnBeliefMCTS,
    R215Observation,
    R215PlanResult,
    R215TurnIdentity,
)

ROOT = Path(__file__).resolve().parents[1]
BELIEF_MCTS_PATH = ROOT / "poke_bot/belief_mcts.py"


@dataclass
class _AuditMCTSResult:
    select: list[int]
    target: object
    sims_run: int
    elapsed_s: float


def _load_belief_mcts_without_torch(monkeypatch):
    """Load the exact search source with only its import-time dependencies faked.

    The audit exercises the real private-search control flow without requiring
    a GPU/Torch installation or a libcg runtime on the test host.
    """

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

    fake_cg_env = types.ModuleType("poke_bot.cg_env")
    fake_cg_env.to_observation = lambda observation: observation
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
    fake_belief.simulator_version = lambda: "audit-simulator"

    fake_blackwell_heads = types.ModuleType("poke_bot.blackwell_heads")
    fake_blackwell_heads.blackwell_strategy_heads_enabled = lambda: False
    fake_blackwell_heads.root_value_bias_from_lethal = lambda _value: 0.0

    fake_adapter = types.ModuleType("poke_bot.matchup_adapter_activation")
    fake_adapter.ShadowMatchupAdapterRouter = object

    fake_mcts = types.ModuleType("poke_bot.mcts")
    fake_mcts.GameClock = object
    fake_mcts.MCTSResult = _AuditMCTSResult

    fake_model = types.ModuleType("poke_bot.model")
    fake_model.TemporalCabtTransformer = object
    fake_model.card_prior_logits_or_uniform = lambda *_args, **_kwargs: None

    fake_replay_import = types.ModuleType("poke_bot.replay_import")
    fake_replay_import.assert_info_set = lambda _obs: None

    fake_targets = types.ModuleType("poke_bot.search_targets")
    fake_targets.build_search_target = lambda combos, visits, _value, **kwargs: SimpleNamespace(
        action_combos=[list(combo) for combo in combos],
        visits=list(visits),
        prior=list(kwargs.get("prior") or []),
        diagnostics=dict(kwargs.get("diagnostics") or {}),
    )
    fake_targets.select_by_visits = lambda combos, visits: list(
        combos[max(range(len(combos)), key=lambda index: (visits[index], -index))]
    )

    for name, module in {
        "torch": fake_torch,
        "poke_bot.batched_infer": fake_batched_infer,
        "poke_bot.belief": fake_belief,
        "poke_bot.blackwell_heads": fake_blackwell_heads,
        "poke_bot.cg_env": fake_cg_env,
        "poke_bot.config": fake_config,
        "poke_bot.features": fake_features,
        "poke_bot.matchup_adapter_activation": fake_adapter,
        "poke_bot.mcts": fake_mcts,
        "poke_bot.model": fake_model,
        "poke_bot.replay_import": fake_replay_import,
        "poke_bot.search_targets": fake_targets,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "poke_bot._r218_belief_mcts_audit"
    spec = importlib.util.spec_from_file_location(module_name, BELIEF_MCTS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, fake_cg_env


def _engine_for_completed_terminal_root(module, fake_cg_env, *, actions, priors, clock):
    root_observation = SimpleNamespace(current=SimpleNamespace(yourIndex=0))
    child_observation = SimpleNamespace(current=SimpleNamespace(yourIndex=0))
    root_state = SimpleNamespace(searchId=1, observation=root_observation)
    child_state = SimpleNamespace(searchId=2, observation=child_observation)
    particle = SimpleNamespace(
        search_inputs={"private": "audit"},
        opponent_deck_digest="audit-deck",
        support_mode="audit",
        support_repairs=0,
    )

    fake_cg_env.to_observation = lambda _observation: root_observation
    fake_cg_env.search_begin = lambda *_args, **_kwargs: root_state
    fake_cg_env.search_step = lambda *_args, **_kwargs: child_state
    fake_cg_env.search_end = lambda: None

    engine = object.__new__(module.BeliefMCTS)
    engine.min_trusted_sims = 1
    engine.particle_count = 2
    engine.max_depth = 1
    engine.puct_c = 1.0
    engine.rng = random.Random(0)
    engine.own_deck = tuple([1] * 60)
    engine.posterior = SimpleNamespace(
        sample_particle=lambda *_args, **_kwargs: particle,
    )
    engine.matchup_shadow_router = None
    engine.leaf_evaluator_source = "audit"
    engine.leaf_evaluator_checkpoint_digest = None
    engine._telemetry_mark = lambda: None
    engine._telemetry_since = lambda _marker: {}
    engine._root_neural_priors = lambda **_kwargs: SimpleNamespace(
        lethal_threat_logit=None,
        prize_race=None,
    )
    engine._terminal_value = lambda _obs, _seat: 0.25
    engine._sample_chance_until_decision = lambda state: (state, 0, ())

    def evaluate(node, _obs, **_kwargs):
        node.network_evaluated = True
        node.factorized = False
        node.total_action_count = len(actions)
        node.edges = [
            module.BeliefEdge(list(action), float(prior))
            for action, prior in zip(actions, priors)
        ]
        return 0.0, True

    engine._evaluate_node_cached = evaluate
    module.information_state_fingerprint = lambda _obs: "audit-root"
    module._raw_observation = lambda _obs: {}
    module.time = SimpleNamespace(perf_counter=clock)
    return engine


def test_minimum_one_completed_backup_returns_best_root_action_at_deadline(
    monkeypatch,
) -> None:
    module, fake_cg_env = _load_belief_mcts_without_torch(monkeypatch)
    ticks = iter((0.0, 0.0, 0.0, 1.0, 1.0))
    engine = _engine_for_completed_terminal_root(
        module,
        fake_cg_env,
        actions=((0,), (1,), (2,)),
        priors=(0.95, 0.033, 0.017),
        clock=lambda: next(ticks),
    )
    engine.convergence_min_sims = 99
    engine.convergence_stable_sims = 99
    engine.convergence_visit_share = 1.0
    engine.convergence_q_margin = 1.0

    result = engine._search_impl(
        {"audit": "root"},
        belief_history=SimpleNamespace(observe=lambda _obs: None),
        root_history_boards=(),
        root_history_previous_actions=(),
        max_sims=50,
        move_time_s=1.0,
    )

    diagnostics = result.target.diagnostics
    assert result.sims_run == 1
    assert result.select == [0]
    assert diagnostics["root_visits"] == diagnostics["value_backups"] == 1
    assert diagnostics["root_action_stable"] is False
    assert diagnostics["stop_reason"] == "valid_move_time_budget"


def test_puct_selects_dominant_prior_then_the_three_percent_challenger(
    monkeypatch,
) -> None:
    module, _fake_cg_env = _load_belief_mcts_without_torch(monkeypatch)
    engine = object.__new__(module.BeliefMCTS)
    engine.puct_c = 1.0
    dominant = module.BeliefEdge([0], 0.95)
    challenger = module.BeliefEdge([1], 0.033)
    tiny = module.BeliefEdge([2], 0.017)
    root = module.BeliefNode(
        fingerprint="root",
        actor=0,
        depth=0,
        edges=[dominant, challenger, tiny],
    )

    assert engine._select_edge(root, root_seat=0) is dominant

    # Once the 95% root line has consumed the early visits, its PUCT bonus is
    # below the unvisited 3.3% challenger, which in turn stays above tiny mass.
    root.visit = dominant.visit = 100
    assert engine._select_edge(root, root_seat=0) is challenger


def test_convergence_receipt_requires_a_completed_backed_up_legal_root(
    monkeypatch,
) -> None:
    module, fake_cg_env = _load_belief_mcts_without_torch(monkeypatch)
    ticks = iter((0.0, 0.0, 0.0, 0.1))
    engine = _engine_for_completed_terminal_root(
        module,
        fake_cg_env,
        actions=((0,),),
        priors=(1.0,),
        clock=lambda: next(ticks),
    )
    engine.convergence_min_sims = 32
    engine.convergence_stable_sims = 16
    engine.convergence_visit_share = 0.9
    engine.convergence_q_margin = 0.05

    result = engine._search_impl(
        {"audit": "root"},
        belief_history=SimpleNamespace(observe=lambda _obs: None),
        root_history_boards=(),
        root_history_previous_actions=(),
        max_sims=50,
        move_time_s=1.0,
    )

    diagnostics = result.target.diagnostics
    receipt = diagnostics["root_stability_receipt"]
    assert result.select == [0]
    assert diagnostics["root_action_stable"] is True
    assert receipt["selected_action"] == [0]
    assert receipt["single_legal_action"] is True
    assert receipt["completed_backups"] == 1
    assert receipt["completed_backups"] == diagnostics["root_visits"]
    assert receipt["completed_backups"] == diagnostics["value_backups"]


def test_later_same_turn_boundary_never_opens_a_second_planner_request() -> None:
    requests = []

    class Planner:
        def plan_turn(self, request):
            requests.append(request)
            return R215PlanResult(selected_action=(1,), sims_run=1)

    ticks = iter((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    controller = R215FullTurnBeliefMCTS(
        Planner(),
        direct_policy=lambda observation: observation.legal_actions[0],
        monotonic=lambda: next(ticks),
    )
    identity = R215TurnIdentity(0, "audit-turn")
    root = R215Observation("root", ((0,), (1,)))
    after_boundary = R215Observation("after-chance", ((0,), (1,)))

    first = controller.act(identity, root)
    later = controller.act(identity, after_boundary, boundary_reason="chance")

    assert first.source == "belief_mcts"
    assert later.source == "direct_policy_fallback"
    assert later.selected_action == (0,)
    assert len(requests) == 1
    assert later.receipt["first_decision_mcts_search_executed"] is False
    assert later.receipt["effective_first_decision_search_allowance_seconds"] == 0.0
    assert later.receipt["fallback_reason"] == "same_turn_boundary_requires_direct_fallback"
