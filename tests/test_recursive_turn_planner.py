"""Unit tests for the lightweight Recursive Turn Planner experiment."""

from __future__ import annotations

import torch
import pytest

from poke_bot.recursive_turn_planner import (
    LatentTransitionDynamics,
    NodeKind,
    ObservationPredicate,
    PersistentTurnMemory,
    PlanExecutor,
    PlanNode,
    RTPConfig,
    RecursiveTurnPlanner,
    SubgoalKind,
    TurnProgram,
    TypedLegalityVerifier,
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
    assert step.repaired is True or executor.repairs_used == 1


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
