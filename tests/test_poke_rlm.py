"""Unit tests for the PokeRLM kit implementation (phases 1–7)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from poke_bot.agent import PolicyAgent
from poke_bot.poke_rlm import (
    PokeRLMConfig,
    PokeRLMController,
    PokeRLMMode,
    PokeRLMModelCore,
    ReasonCode,
    TurnComputeBudget,
    config_for_profile,
    load_poke_rlm_config_from_env,
    resolve_selector,
    round_trip_plan,
    turn_plan_to_json,
)
from poke_bot.poke_rlm.legal_action import (
    ActionSelector,
    LegalAction,
    legal_actions_from_select,
)
from poke_bot.poke_rlm.observation import (
    build_deployment_observation,
    reject_hidden_fields,
)
from poke_bot.poke_rlm.plan_ir import (
    ObjectiveCode,
    PLAN_IR_SCHEMA_VERSION,
    PrimitiveAction,
    SequenceNode,
    StopNode,
    StopReason,
    SubgoalNode,
    TurnPlan,
    validate_plan_json_schema,
    validate_plan_static,
)
from poke_bot.poke_rlm.recursion import refine_plan, shared_recursive_cell_identity
from poke_bot.poke_rlm.router import Route, choose_route
from poke_bot.poke_rlm.executor import PlanExecutor
from poke_bot.poke_rlm.training import (
    HardStateCurriculum,
    build_labels_from_trace,
    compute_poke_rlm_losses,
)
from poke_bot.poke_rlm.training.traces import TurnPlanTrace


def _simple_obs(*, with_hidden: bool = False, n_opts: int = 3) -> dict:
    opp: dict = {"hand": None, "prizeCards": 6}
    if with_hidden:
        opp["hand"] = [1, 2, 3]
    return {
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "players": [
                {"hand": [10, 11], "prizeCards": 6},
                opp,
            ],
        },
        "select": {
            "type": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14, "cardId": 100 + i} for i in range(n_opts)],
        },
    }


@pytest.mark.unit
def test_config_defaults_preserve_behavior() -> None:
    cfg = PokeRLMConfig()
    assert cfg.enabled is False
    assert cfg.mode is PokeRLMMode.DISABLED
    assert cfg.runs_planner is False
    assert cfg.selects_actions is False
    assert "selects_actions" in cfg.inventory()


@pytest.mark.unit
def test_env_loader_enables_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POKEBOT_POKE_RLM_ENABLED", "1")
    monkeypatch.setenv("POKEBOT_POKE_RLM_MODE", "shadow")
    monkeypatch.setenv("POKEBOT_POKE_RLM_PROFILE", "pure_rl_96")
    cfg = load_poke_rlm_config_from_env()
    assert cfg.enabled is True
    assert cfg.mode is PokeRLMMode.SHADOW
    assert cfg.shadow_only is True
    assert cfg.selects_actions is False
    assert cfg.d_model == 96


@pytest.mark.unit
def test_hidden_field_rejection() -> None:
    assert reject_hidden_fields(_simple_obs()) is ReasonCode.OK
    assert reject_hidden_fields(_simple_obs(with_hidden=True)) is ReasonCode.HIDDEN_FIELD_REJECTED
    dep, reason = build_deployment_observation(_simple_obs(with_hidden=True))
    assert dep is None
    assert reason is ReasonCode.HIDDEN_FIELD_REJECTED
    dep, reason = build_deployment_observation(_simple_obs())
    assert reason is ReasonCode.OK
    assert dep is not None
    assert dep.observation_hash.startswith("sha256:")


@pytest.mark.unit
def test_selector_resolve_under_reorder() -> None:
    obs = _simple_obs(n_opts=3)
    legal = legal_actions_from_select(obs)
    target = legal[1]
    selector = ActionSelector(
        fingerprint=target.fingerprint,
        planned_option_index_path=(0,),  # stale index on purpose
    )
    reordered = (legal[2], legal[0], legal[1])
    resolved, reason = resolve_selector(selector, reordered)
    assert reason is ReasonCode.OK
    assert resolved is not None
    assert resolved.fingerprint == target.fingerprint


@pytest.mark.unit
def test_budget_enforcement() -> None:
    budget = TurnComputeBudget(max_model_calls=2, max_simulator_calls=1)
    assert budget.consume_model_call() is ReasonCode.OK
    assert budget.consume_model_call() is ReasonCode.OK
    assert budget.consume_model_call() is ReasonCode.BUDGET_MODEL_CALLS
    assert budget.consume_simulator_call() is ReasonCode.OK
    assert budget.consume_simulator_call() is ReasonCode.BUDGET_SIMULATOR_CALLS


@pytest.mark.unit
def test_plan_schema_round_trip() -> None:
    obs = _simple_obs()
    legal = legal_actions_from_select(obs)
    # Fix GoalCode properly
    from poke_bot.poke_rlm.plan_ir import GoalCode

    plan = TurnPlan(
        plan_id="rt-1",
        schema_version=PLAN_IR_SCHEMA_VERSION,
        objective=ObjectiveCode.TAKE_KNOCKOUT,
        root=SequenceNode(
            children=(
                PrimitiveAction(
                    selector=ActionSelector(
                        fingerprint=legal[0].fingerprint,
                        planned_option_index_path=legal[0].option_index_path,
                    ),
                    planned_option_index=legal[0].option_index_path,
                    expected_group="main",
                    on_unresolvable=StopNode(reason=StopReason.FALLBACK_REQUIRED),
                ),
                SubgoalNode(
                    goal_code=GoalCode.DEVELOP_BOARD,
                    fallback=StopNode(reason=StopReason.FALLBACK_REQUIRED),
                ),
                StopNode(reason=StopReason.END_TURN),
            )
        ),
    )
    assert validate_plan_static(plan) is ReasonCode.OK
    payload = turn_plan_to_json(plan)
    assert validate_plan_json_schema(payload) is ReasonCode.OK
    restored = round_trip_plan(plan)
    assert restored.plan_id == plan.plan_id
    assert restored.objective is plan.objective
    assert restored.first_primitive() is not None


@pytest.mark.unit
def test_parallel_decoder_scores_all_actions_once() -> None:
    cfg = config_for_profile("pure_rl_96")
    core = PokeRLMModelCore(cfg)
    core.eval()
    b, n, d = 1, 5, cfg.d_model
    state = torch.randn(b, d)
    opts = torch.randn(b, n, d)
    mask = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        heads = core.score_actions(state, opts, legal_mask=mask)
    assert heads.policy_logits.shape == (1, 5)
    assert torch.isneginf(heads.policy_logits[0, 3])
    assert heads.q_quantiles.shape == (1, 5, cfg.q_quantiles)
    assert heads.successor.shape == (1, 5, cfg.successor_dim)
    inv = core.parameter_inventory()
    assert inv["total"] > 0
    assert inv["decoder"] > 0


@pytest.mark.unit
def test_latent_dynamics_shapes() -> None:
    cfg = config_for_profile("pure_rl_96")
    core = PokeRLMModelCore(cfg)
    z = torch.randn(1, cfg.d_model)
    a = torch.randn(4, cfg.d_model)
    out = core.dynamics(z, a)
    assert out["next_latent"].shape == (4, cfg.d_model)
    assert out["structured_delta"].shape == (4, 4)


@pytest.mark.unit
def test_root_plan_proposer_and_static_validity() -> None:
    cfg = config_for_profile("pure_rl_96", root_plan_candidates=4)
    core = PokeRLMModelCore(cfg)
    obs = _simple_obs(n_opts=4)
    legal = legal_actions_from_select(obs)
    state = torch.randn(1, cfg.d_model)
    opts = torch.randn(1, len(legal), cfg.d_model)
    heads = core.score_actions(state, opts)
    plans = core.propose_root_plans(state, legal, heads, observation_hash="sha256:x")
    assert plans
    assert all(validate_plan_static(p) is ReasonCode.OK for p in plans)


@pytest.mark.unit
def test_shared_recursive_cell_identity_across_depth() -> None:
    cfg = config_for_profile("pure_rl_96", max_depth=2, max_neural_planner_calls_per_turn=8)
    core = PokeRLMModelCore(cfg)
    cell_id = shared_recursive_cell_identity(core)
    obs = _simple_obs(n_opts=4)
    legal = legal_actions_from_select(obs)
    state = torch.randn(1, cfg.d_model)
    opts = torch.randn(1, len(legal), cfg.d_model)
    heads = core.score_actions(state, opts)
    plans = core.propose_root_plans(state, legal, heads)
    budget = TurnComputeBudget(
        max_depth=2,
        max_model_calls=8,
        max_nodes=32,
        max_subgoals=8,
        max_simulator_calls=16,
    )
    refined, reason = refine_plan(
        core,
        plans[0],
        state_vec=state,
        option_hidden=opts,
        legal=legal,
        budget=budget,
    )
    assert reason is ReasonCode.OK
    assert shared_recursive_cell_identity(core) == cell_id
    assert refined.recursion_depth >= 0


@pytest.mark.unit
def test_executor_branch_and_repair() -> None:
    obs = _simple_obs(n_opts=2)
    legal = legal_actions_from_select(obs)
    plan = TurnPlan(
        plan_id="exec",
        schema_version=PLAN_IR_SCHEMA_VERSION,
        objective=ObjectiveCode.DEVELOP_BOARD,
        root=PrimitiveAction(
            selector=ActionSelector(fingerprint="missing"),
            planned_option_index=(9,),
            expected_group="main",
            on_unresolvable=None,
        ),
    )
    repairs = {"n": 0}

    def repair_fn(p, leg):
        repairs["n"] += 1
        return TurnPlan(
            plan_id="repaired",
            schema_version=PLAN_IR_SCHEMA_VERSION,
            objective=ObjectiveCode.DEVELOP_BOARD,
            root=PrimitiveAction(
                selector=ActionSelector(
                    fingerprint=leg[0].fingerprint,
                    planned_option_index_path=leg[0].option_index_path,
                ),
                planned_option_index=leg[0].option_index_path,
                expected_group="main",
            ),
        )

    ex = PlanExecutor(repair_fn=repair_fn, repair_budget=1)
    ex.load(plan)
    step = ex.next_action(legal)
    assert repairs["n"] == 1
    assert step.repaired is True
    assert step.action == legal[0].option_index_path


@pytest.mark.unit
def test_router_trivial_and_recursive() -> None:
    cfg = PokeRLMConfig(complexity_option_threshold=3, complexity_entropy_threshold=0.1)
    d = choose_route(cfg, n_legal=1)
    assert d.route is Route.DIRECT
    logits = torch.zeros(8)
    d2 = choose_route(cfg, n_legal=8, policy_logits=logits)
    assert d2.route is Route.RECURSIVE


@pytest.mark.unit
def test_controller_disabled_and_shadow() -> None:
    obs = _simple_obs(n_opts=3)
    legal = legal_actions_from_select(obs)
    state = torch.randn(96)
    opts = torch.randn(len(legal), 96)
    disabled = PokeRLMController(PokeRLMConfig())
    r0 = disabled.plan_decision(
        obs, state_vec=state, option_hidden=opts, fallback_action=(0,)
    )
    assert r0.reason is ReasonCode.DISABLED
    assert r0.used_for_selection is False
    assert r0.action == (0,)

    shadow_cfg = PokeRLMConfig(enabled=True, mode=PokeRLMMode.SHADOW, d_model=96)
    shadow = PokeRLMController(shadow_cfg)
    r1 = shadow.plan_decision(
        obs,
        state_vec=state,
        option_hidden=opts,
        legal_combos=[a.option_index_path for a in legal],
        fallback_action=(1,),
    )
    assert r1.used_for_selection is False
    assert r1.action == (1,)
    assert r1.trace.enabled is True

    active_cfg = PokeRLMConfig(enabled=True, mode=PokeRLMMode.ACTIVE, d_model=96)
    active = PokeRLMController(active_cfg)
    r2 = active.plan_decision(
        obs,
        state_vec=state,
        option_hidden=opts,
        legal_combos=[a.option_index_path for a in legal],
        fallback_action=(1,),
    )
    assert r2.used_for_selection is True
    assert r2.action in {a.option_index_path for a in legal}


@pytest.mark.unit
def test_controller_rejects_hidden_without_sim_calls() -> None:
    cfg = PokeRLMConfig(enabled=True, mode=PokeRLMMode.ACTIVE, d_model=96)
    ctl = PokeRLMController(cfg)
    r = ctl.plan_decision(
        _simple_obs(with_hidden=True),
        state_vec=torch.zeros(96),
        option_hidden=torch.zeros(1, 96),
        fallback_action=(0,),
    )
    assert r.reason is ReasonCode.HIDDEN_FIELD_REJECTED
    assert ctl.budget.simulator_calls == 0


@pytest.mark.unit
def test_training_losses_and_hard_states() -> None:
    cfg = config_for_profile("pure_rl_96")
    core = PokeRLMModelCore(cfg)
    obs = _simple_obs(n_opts=3)
    legal = legal_actions_from_select(obs)
    state = torch.randn(1, cfg.d_model)
    opts = torch.randn(1, 3, cfg.d_model)
    heads = core.score_actions(state, opts)
    plans = core.propose_root_plans(state, legal, heads)
    trace = TurnPlanTrace(
        battle_id="b1",
        turn_index=2,
        side_id="0",
        observation_fingerprint="sha256:x",
        legal_action_fingerprints=[a.fingerprint for a in legal],
        chosen_action_fingerprint=legal[0].fingerprint,
        plan=plans[0],
        mode="shadow",
        route="root",
        stop_reason="ok",
    )
    labels = build_labels_from_trace(trace=trace, chosen_action_index=0)
    assert labels.commit_indices
    route_logits = torch.randn(3)
    recurse_logits = torch.randn(1)
    bundle = compute_poke_rlm_losses(
        action_logits=heads.policy_logits[0],
        route_logits=route_logits,
        recurse_logits=recurse_logits,
        labels=labels,
        predicted_next_latent=torch.randn(1, cfg.d_model),
        target_next_latent=torch.randn(1, cfg.d_model),
    )
    assert torch.isfinite(bundle.total)
    assert "action" in bundle.as_dict()

    curriculum = HardStateCurriculum(capacity=2)
    assert curriculum.consider(battle_id="b1", turn_index=1, reason="fork", score=0.5)
    assert curriculum.consider(battle_id="b2", turn_index=2, reason="fork", score=0.9)
    assert curriculum.consider(battle_id="b3", turn_index=3, reason="fork", score=0.1) is False
    assert curriculum.sample_ids(1)[0][0] == "b2"


@pytest.mark.unit
def test_policy_agent_poke_rlm_disabled_by_default() -> None:
    model = SimpleNamespace(
        d_model=96,
        eval=lambda: None,
        parameters=lambda: iter((torch.zeros(1),)),
    )
    agent = PolicyAgent(
        model=model,  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        use_recursive_turn_planner=False,
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent.poke_rlm_config is not None
    assert agent.poke_rlm_config.enabled is False
    assert agent._poke_rlm_bridge is None


@pytest.mark.unit
def test_policy_agent_poke_rlm_shadow_inits_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_POKE_RLM_ENABLED", "1")
    monkeypatch.setenv("POKEBOT_POKE_RLM_MODE", "shadow")
    model = SimpleNamespace(
        d_model=96,
        eval=lambda: None,
        parameters=lambda: iter((torch.zeros(1),)),
    )
    agent = PolicyAgent(
        model=model,  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        use_recursive_turn_planner=False,
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent.poke_rlm_config.shadow_only is True
    assert agent._poke_rlm_bridge is not None
    agent._poke_rlm_bridge.controller.active_turn_key = (0, 1)  # type: ignore[union-attr]
    agent.reset_game()
    assert agent._poke_rlm_bridge.controller.active_turn_key == (-1, -1)  # type: ignore[union-attr]
