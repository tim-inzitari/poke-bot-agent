"""Unit tests for the lightweight Recursive Turn Planner experiment."""

from __future__ import annotations

import torch
import pytest

from poke_bot.model import ActionConditionedLatentLookahead
from poke_bot.recursive_turn_planner import (
    GLOBAL_TRANSFORMER,
    PURE_RL,
    PURE_RL_R197,
    LatentTransitionDynamics,
    LookaheadBackedDynamics,
    NodeKind,
    ObservationPredicate,
    PersistentTurnMemory,
    PlanExecutor,
    PlanNode,
    RTPConfig,
    RTP_MAX_AUTHORIZED_NEURAL_PASSES,
    RecursiveTurnPlanner,
    SubgoalKind,
    TurnProgram,
    TypedLegalityVerifier,
    VERIFY_ABLATONS,
    get_profile,
    profile_inventory,
    required_recursive_passes,
)


@pytest.mark.unit
def test_plan_node_requires_typed_structure() -> None:
    with pytest.raises(ValueError):
        PlanNode(kind=NodeKind.PRIMITIVE)
    with pytest.raises(ValueError):
        PlanNode(kind=NodeKind.BRANCH, predicate=ObservationPredicate("x"))
    node = PlanNode(
        kind=NodeKind.SEQUENCE,
        children=(
            PlanNode(kind=NodeKind.PRIMITIVE, action=(1,)),
            PlanNode(kind=NodeKind.END_TURN),
        ),
    )
    assert node.first_action() == (1,)
    assert node.flatten_primitives() == ((1,),)


@pytest.mark.unit
def test_encode_once_memory_and_incremental_update() -> None:
    planner = RecursiveTurnPlanner(RTPConfig(d_model=16, dynamics_width=32))
    state = torch.randn(16)
    memory = planner.encode_memory(
        state,
        legal_actions=[(0,), (1, 2)],
        value_estimate=0.25,
        remaining_resources={"supporter": True},
    )
    assert memory.d_model == 16
    assert memory.encode_pass_count == 1
    before = memory.snapshot_latent().clone()
    memory.update_observation(
        observations={"target_found": True},
        latent_delta=torch.ones(16) * 0.1,
    )
    assert memory.observations["target_found"] is True
    assert not torch.equal(memory.snapshot_latent(), before)


@pytest.mark.unit
def test_complexity_gate_uses_direct_policy_on_simple_positions() -> None:
    cfg = RTPConfig(
        d_model=16,
        dynamics_width=32,
        complexity_option_threshold=8,
        complexity_entropy_threshold=10.0,
        num_plan_candidates=2,
        max_neural_passes=4,
    )
    planner = RecursiveTurnPlanner(cfg)
    # Force complexity head toward "simple".
    with torch.no_grad():
        planner.complexity_head[-1].bias.fill_(-10.0)
    memory = planner.encode_memory(
        torch.zeros(16),
        legal_actions=[(0,), (1,), (2,)],
    )
    logits = torch.tensor([0.1, 2.0, 0.2])
    decision = planner.plan_turn(memory, policy_logits=logits, force_recurse=False)
    assert decision.used_direct_policy is True
    assert decision.mode == "direct_policy"
    assert decision.action == (1,)


@pytest.mark.unit
def test_recursive_plan_proposes_legal_typed_program() -> None:
    cfg = RTPConfig(
        d_model=16,
        dynamics_width=32,
        num_plan_candidates=3,
        max_recursion_depth=2,
        max_neural_passes=8,
        max_plan_length=8,
        complexity_option_threshold=1,
        complexity_entropy_threshold=0.0,
    )
    planner = RecursiveTurnPlanner(cfg)
    legal = [(0,), (1,), (2,), (3,), (4,)]
    memory = planner.encode_memory(torch.randn(16), legal_actions=legal)
    decision = planner.plan_turn(memory, force_recurse=True)
    assert decision.mode == "recursive_plan"
    assert decision.program is not None
    assert decision.action in set(legal)
    assert decision.neural_passes <= cfg.max_neural_passes
    report = TypedLegalityVerifier().validate_program(
        decision.program, memory.legal_actions
    )
    assert report.valid


@pytest.mark.unit
def test_latent_dynamics_is_action_conditioned() -> None:
    dyn = LatentTransitionDynamics(8, width=16)
    state = torch.randn(1, 8)
    embed_a = torch.zeros(1, 8)
    embed_b = torch.ones(1, 8)
    out_a = dyn(state, embed_a)["next_latent"]
    out_b = dyn(state, embed_b)["next_latent"]
    assert not torch.equal(out_a, out_b)
    rollout = dyn.rollout_program_value(
        state[0],
        ((0,), (1, 2)),
        max_horizon=2,
    )
    assert "value" in rollout and "uncertainty" in rollout


@pytest.mark.unit
def test_legality_prunes_illegal_primitives() -> None:
    verifier = TypedLegalityVerifier()
    program = TurnProgram(
        root=PlanNode(kind=NodeKind.PRIMITIVE, action=(9,))
    )
    report = verifier.validate_program(program, legal_actions=((0,), (1,)))
    assert report.valid is False
    assert report.reason == "illegal_primitive"

    mixed = TurnProgram(
        root=PlanNode(
            kind=NodeKind.SEQUENCE,
            children=(
                PlanNode(kind=NodeKind.PRIMITIVE, action=(9,)),
                PlanNode(kind=NodeKind.PRIMITIVE, action=(0,)),
                PlanNode(kind=NodeKind.END_TURN),
            ),
        )
    )
    mixed_report = verifier.validate_node(mixed.root, legal_actions=((0,), (1,)))
    assert mixed_report.valid is True
    assert (0,) in mixed_report.kept_actions
    assert (9,) not in mixed_report.kept_actions


@pytest.mark.unit
def test_executor_follows_branch_and_persists_plan() -> None:
    program = TurnProgram(
        root=PlanNode(
            kind=NodeKind.SEQUENCE,
            children=(
                PlanNode(kind=NodeKind.PRIMITIVE, action=(0,)),
                PlanNode(
                    kind=NodeKind.BRANCH,
                    predicate=ObservationPredicate("target_found", True),
                    children=(
                        PlanNode(kind=NodeKind.PRIMITIVE, action=(1,)),
                        PlanNode(kind=NodeKind.PRIMITIVE, action=(2,)),
                    ),
                ),
                PlanNode(kind=NodeKind.END_TURN),
            ),
        )
    )
    memory = PersistentTurnMemory(
        H=torch.zeros(8),
        legal_actions=((0,), (1,), (2,)),
    )
    executor = PlanExecutor(RTPConfig(d_model=8, repair_budget=1))
    executor.load(program)

    step1 = executor.next_action(memory)
    assert step1.action == (0,)
    assert step1.done is False

    step2 = executor.next_action(
        memory, observations={"target_found": True}
    )
    assert step2.action == (1,)

    step3 = executor.next_action(memory)
    assert step3.done is True


@pytest.mark.unit
def test_executor_repairs_when_plan_becomes_illegal() -> None:
    bad = TurnProgram(
        root=PlanNode(kind=NodeKind.PRIMITIVE, action=(9,)),
        plan_id="bad",
    )
    good = TurnProgram(
        root=PlanNode(kind=NodeKind.PRIMITIVE, action=(1,)),
        plan_id="good",
    )
    memory = PersistentTurnMemory(
        H=torch.zeros(8),
        legal_actions=((1,),),
    )

    def repair_fn(_memory: PersistentTurnMemory, _program: TurnProgram) -> TurnProgram:
        return good

    executor = PlanExecutor(
        RTPConfig(d_model=8, repair_budget=1),
        repair_fn=repair_fn,
    )
    executor.load(bad)
    step = executor.next_action(memory)
    assert step.action == (1,)
    assert step.repaired is True
    assert executor.repairs_used == 1


@pytest.mark.unit
def test_executor_illegal_extraction_without_callback_is_not_repaired() -> None:
    mixed = TurnProgram(
        root=PlanNode(
            kind=NodeKind.SEQUENCE,
            children=(
                PlanNode(kind=NodeKind.PRIMITIVE, action=(9,)),
                PlanNode(kind=NodeKind.PRIMITIVE, action=(1,)),
            ),
        ),
        plan_id="mixed-no-repair",
    )
    memory = PersistentTurnMemory(
        H=torch.zeros(8),
        legal_actions=((1,),),
    )
    executor = PlanExecutor(RTPConfig(d_model=8, repair_budget=1))
    executor.load(mixed)

    step = executor.next_action(memory)

    assert step.action is None
    assert step.done is True
    assert step.repaired is False
    assert step.reason == "extracted_action_illegal"
    assert executor.repairs_used == 0


@pytest.mark.unit
def test_executor_illegal_extraction_with_callback_reports_actual_repair() -> None:
    mixed = TurnProgram(
        root=PlanNode(
            kind=NodeKind.SEQUENCE,
            children=(
                PlanNode(kind=NodeKind.PRIMITIVE, action=(9,)),
                PlanNode(kind=NodeKind.PRIMITIVE, action=(1,)),
            ),
        ),
        plan_id="mixed-repair",
    )
    good = TurnProgram(
        root=PlanNode(kind=NodeKind.PRIMITIVE, action=(1,)),
        plan_id="actual-repair",
    )
    memory = PersistentTurnMemory(
        H=torch.zeros(8),
        legal_actions=((1,),),
    )
    executor = PlanExecutor(
        RTPConfig(d_model=8, repair_budget=1),
        repair_fn=lambda _memory, _program: good,
    )
    executor.load(mixed)

    step = executor.next_action(memory)

    assert step.action == (1,)
    assert step.repaired is True
    assert step.reason == "ok"
    assert executor.repairs_used == 1


@pytest.mark.unit
def test_neural_pass_budget_is_hard_capped() -> None:
    cfg = RTPConfig(
        d_model=8,
        dynamics_width=16,
        max_neural_passes=1,
        num_plan_candidates=2,
    )
    planner = RecursiveTurnPlanner(cfg)
    memory = planner.encode_memory(torch.zeros(8), legal_actions=[(0,), (1,)])
    planner.reset_pass_counter()
    planner._consume_pass()
    with pytest.raises(RuntimeError, match="neural-pass budget"):
        planner._consume_pass()


@pytest.mark.unit
def test_subgoal_kinds_are_explicit_vocabulary() -> None:
    assert SubgoalKind.FIND_RESOURCE.value == "find_resource"
    node = PlanNode(
        kind=NodeKind.SUBGOAL,
        subgoal=SubgoalKind.MAXIMIZE_DRAW,
        subgoal_label="maximize_draw",
    )
    assert node.subgoal is SubgoalKind.MAXIMIZE_DRAW


@pytest.mark.unit
def test_sizing_profiles_match_in_repo_parents() -> None:
    assert GLOBAL_TRANSFORMER.d_model == 256
    assert GLOBAL_TRANSFORMER.dynamics_width == 512
    assert PURE_RL.d_model == 96
    assert PURE_RL.dynamics_width == 192
    assert GLOBAL_TRANSFORMER.max_neural_passes == 4
    assert PURE_RL.max_neural_passes == 4
    assert GLOBAL_TRANSFORMER.complexity_option_threshold == 8
    assert PURE_RL.complexity_option_threshold == 8
    cfg = get_profile("pure_rl").to_config(online_sim_verify_budget=8)
    assert cfg.d_model == 96
    assert cfg.online_sim_verify_budget == 8
    inv = profile_inventory()
    assert inv["reuse_targets"]["pure_rl_d_model"] == 96
    assert VERIFY_ABLATONS["legacy_mcts_75"] == 75
    default = RecursiveTurnPlanner()
    assert default.config.d_model == 256
    assert default.config.dynamics_width == 512
    assert default.inventory()["sizing_profile"] == "global_transformer"


@pytest.mark.unit
def test_r197_profile_completes_normal_and_forced_recursive_paths() -> None:
    cfg = get_profile("pure_rl_r197").to_config()
    assert cfg == PURE_RL_R197.to_config()
    assert (cfg.d_model, cfg.num_plan_candidates, cfg.max_recursion_depth) == (
        96,
        4,
        2,
    )
    assert cfg.max_neural_passes == 256
    assert required_recursive_passes(cfg) == 6
    assert required_recursive_passes(cfg, force_recurse=True) == 5

    planner = RecursiveTurnPlanner(cfg)
    legal = [(index,) for index in range(8)]
    normal_memory = planner.encode_memory(torch.randn(96), legal_actions=legal)
    normal = planner.plan_turn(
        normal_memory,
        policy_logits=torch.zeros(len(legal)),
    )
    assert normal.mode == "recursive_plan"
    assert normal.neural_passes == 6

    forced_memory = planner.encode_memory(torch.randn(96), legal_actions=legal)
    forced = planner.plan_turn(forced_memory, force_recurse=True)
    assert forced.mode == "recursive_plan"
    assert forced.neural_passes == 5


@pytest.mark.unit
def test_neural_pass_hard_ceiling_is_bounded() -> None:
    assert (
        RTPConfig(
            max_neural_passes=RTP_MAX_AUTHORIZED_NEURAL_PASSES
        ).max_neural_passes
        == 256
    )
    with pytest.raises(ValueError, match="authorized hard ceiling"):
        RTPConfig(max_neural_passes=RTP_MAX_AUTHORIZED_NEURAL_PASSES + 1)


@pytest.mark.unit
def test_trivial_decisions_skip_recursion() -> None:
    planner = RecursiveTurnPlanner(
        RTPConfig(d_model=16, dynamics_width=32, skip_trivial_decisions=True)
    )
    memory = planner.encode_memory(torch.zeros(16), legal_actions=[(3,)])
    decision = planner.plan_turn(
        memory,
        policy_logits=torch.tensor([1.0]),
    )
    assert decision.used_direct_policy is True
    assert decision.diagnostics["complexity_gate"]["trivial"] is True


@pytest.mark.unit
def test_option_hidden_preferred_over_action_id_hash() -> None:
    dyn = LatentTransitionDynamics(8, width=16, prefer_option_hidden=True)
    actions = ((0,), (1,))
    option_hidden = torch.randn(2, 8)
    scored = dyn.score_actions(torch.randn(8), actions, option_hidden=option_hidden)
    assert scored["action_embed_source"] == "option_hidden"
    hashed = dyn.score_actions(torch.randn(8), actions, option_hidden=None)
    assert hashed["action_embed_source"] == "action_id_hash"

    planner = RecursiveTurnPlanner(
        RTPConfig(
            d_model=8,
            dynamics_width=16,
            num_plan_candidates=2,
            max_neural_passes=6,
            prefer_option_hidden=True,
        )
    )
    legal = [(0,), (1,), (2,)]
    memory = planner.encode_memory(
        torch.randn(8),
        legal_actions=legal,
        option_hidden=torch.randn(3, 8),
    )
    decision = planner.plan_turn(memory, force_recurse=True)
    assert decision.mode == "recursive_plan"
    assert decision.diagnostics["used_option_hidden"] is True


@pytest.mark.unit
def test_lookahead_backed_dynamics_reuses_existing_module() -> None:
    lookahead = ActionConditionedLatentLookahead(16, width=32, policy_aid_cap=0.25)
    dyn = LookaheadBackedDynamics(lookahead, d_model=16)
    state = torch.randn(1, 16)
    options = torch.randn(3, 16)
    out = dyn.score_actions(
        state,
        ((0,), (1,), (2,)),
        option_hidden=options,
    )
    assert out["action_embed_source"] == "option_hidden"
    assert isinstance(out["value"], torch.Tensor)
    assert out["value"].shape == (3,)
    inv = dyn.inventory()
    assert inv["backed_by"] == "ActionConditionedLatentLookahead"
    assert inv["lookahead_inventory"]["mcts_allowed"] is False
