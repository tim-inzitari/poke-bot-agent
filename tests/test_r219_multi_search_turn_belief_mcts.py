from __future__ import annotations

from types import SimpleNamespace

import poke_bot.r219_multi_search_turn_belief_mcts as r219_module
from poke_bot.r215_full_turn_belief_mcts import (
    BeliefMCTSPlannerAdapter,
    R215CachedBranchStep,
    R215GameClock,
    R215Observation,
    R215PlanRequest,
    R215PlanResult,
    R215TurnIdentity,
)
from poke_bot.r219_multi_search_turn_belief_mcts import (
    R219FiniteChanceOutcome,
    R219FiniteChanceReceipt,
    R219MultiSearchTurnBeliefMCTS,
    R219PolicyTurnBridge,
    R219TimingConfig,
    r219_plan_result_from_mcts_result,
)


class _Ticks:
    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("controller consumed more clock values than expected")
        return self.values.pop(0)


class _Planner:
    def __init__(self, *results: R215PlanResult) -> None:
        self.results = list(results)
        self.requests: list[R215PlanRequest] = []

    def plan_turn(self, request: R215PlanRequest) -> R215PlanResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("unexpected planner request")
        return self.results.pop(0)


def _obs(name: str) -> R215Observation:
    return R215Observation(f"public:{name}", ((0,), (1,), (2,)))


def _cached(
    observation: R215Observation,
    action: tuple[int, ...],
    *,
    finite_outcome_id: str | None = None,
) -> R215CachedBranchStep:
    return R215CachedBranchStep(
        expected_public_observation_fingerprint=observation.public_observation_fingerprint,
        expected_legal_action_order_fingerprint=observation.legal_action_order_fingerprint,
        selected_action=action,
        requires_verified_finite_chance=finite_outcome_id is not None,
        finite_chance_outcome_id=finite_outcome_id,
    )


def _controller(planner: _Planner, ticks: _Ticks, **kwargs) -> R219MultiSearchTurnBeliefMCTS:
    return R219MultiSearchTurnBeliefMCTS(
        planner,
        direct_policy=lambda observation: observation.legal_actions[0],
        monotonic=ticks,
        **kwargs,
    )


def test_r219_researches_at_cache_endpoint_from_one_shared_45_second_pool() -> None:
    planner = _Planner(
        R215PlanResult(selected_action=(1,), sims_run=1),
        R215PlanResult(selected_action=(2,), sims_run=1),
    )
    controller = _controller(planner, _Ticks(0.0, 4.0, 4.0, 4.0, 9.0, 9.0))
    identity = R215TurnIdentity(0, "turn")

    first = controller.act(identity, _obs("first"))
    second = controller.act(identity, _obs("endpoint"))

    assert first.source == second.source == "belief_mcts"
    assert len(planner.requests) == 2
    assert planner.requests[0].search_segment_index == 1
    assert planner.requests[0].search_boundary_reason == "first_meaningful_choice"
    assert planner.requests[0].effective_search_allowance_seconds == 15.0
    assert planner.requests[1].search_segment_index == 2
    assert planner.requests[1].search_boundary_reason == "validated_cached_plan_endpoint"
    assert planner.requests[1].turn_pool_remaining_seconds == 41.0
    assert planner.requests[1].effective_search_allowance_seconds == 15.0
    assert second.receipt["search_segments_this_turn"] == 2
    assert second.receipt["later_research_count_this_turn"] == 1
    assert second.receipt["rebuild_searches_this_turn"] == 1
    assert second.receipt["rebuild_count_and_reasons"] == {
        "validated_cached_plan_endpoint": 1
    }
    assert second.receipt["planner_seconds_used_this_turn"] == 9.0
    assert second.receipt["planner_seconds_residual_this_turn"] == 36.0
    assert second.receipt["only_first_segment_search"] is False


def test_r219_valid_deterministic_cache_hop_does_not_open_a_second_search() -> None:
    next_observation = _obs("cached")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached(next_observation, (2,)),),
        )
    )
    controller = _controller(planner, _Ticks(0.0, 3.0, 3.0, 3.0, 3.5, 3.5))
    identity = R215TurnIdentity(0, "turn")

    controller.act(identity, _obs("root"))
    cached = controller.act(identity, next_observation)

    assert cached.source == "deterministic_cached_branch"
    assert cached.selected_action == (2,)
    assert len(planner.requests) == 1
    assert cached.receipt["search_segments_this_turn"] == 1
    assert cached.receipt["cache_hops_this_turn"] == 1
    assert cached.receipt["cache_only_later_step"] is True
    assert cached.receipt["cache_only_later_steps_this_turn"] is True
    assert cached.receipt["only_first_segment_search"] is True


def test_r219_chance_boundary_invalidates_only_affected_cache_then_researches() -> None:
    after_chance = _obs("after-chance")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached(after_chance, (2,)),),
        ),
        R215PlanResult(selected_action=(2,), sims_run=1),
    )
    controller = _controller(planner, _Ticks(0.0, 2.0, 2.0, 2.0, 4.0, 4.0))
    identity = R215TurnIdentity(0, "turn")

    controller.act(identity, _obs("root"))
    rebuilt = controller.act(identity, after_chance, boundary_reason="chance")

    assert rebuilt.source == "belief_mcts"
    assert len(planner.requests) == 2
    assert planner.requests[1].search_boundary_reason == "chance_boundary"
    assert rebuilt.receipt["cached_branch_invalidations_and_reasons"] == {"chance": 1}
    assert rebuilt.receipt["rebuild_count_and_reasons"] == {"chance_boundary": 1}
    assert rebuilt.receipt["later_research_count"] == 1
    assert rebuilt.receipt["sampled_or_opaque_chance_boundaries"] == 1


def test_r219_pool_exhaustion_uses_direct_policy_without_abort_or_new_search() -> None:
    planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    timing = R219TimingConfig(
        default_turn_pool_seconds=15.0,
        per_operation_ceiling_seconds=15.0,
        first_decision_search_ceiling_seconds=15.0,
        later_decision_search_ceiling_seconds=15.0,
    )
    controller = _controller(
        planner,
        _Ticks(0.0, 15.0, 15.0, 15.0, 15.0, 15.0),
        timing=timing,
        game_clock=R215GameClock(),
    )
    identity = R215TurnIdentity(0, "turn")

    controller.act(identity, _obs("root"))
    exhausted = controller.act(identity, _obs("endpoint"))

    assert exhausted.source == "direct_policy_fallback"
    assert exhausted.selected_action == (0,)
    assert exhausted.receipt["fallback_reason"] == "turn_or_game_clock_exhausted"
    assert exhausted.receipt["search_segments_this_turn"] == 1
    assert len(planner.requests) == 1


def _coin_receipt(heads: R215Observation) -> R219FiniteChanceReceipt:
    return R219FiniteChanceReceipt(
        engine_force_enumeration_receipt_id="engine:coin:receipt",
        engine_verified_exact_enumeration=True,
        outcomes=(
            R219FiniteChanceOutcome(
                "heads",
                1,
                2,
                heads.public_observation_fingerprint,
                heads.legal_action_order_fingerprint,
            ),
            R219FiniteChanceOutcome("tails", 1, 2, "public:tails", "legal:tails"),
        ),
        realized_outcome_id="heads",
    )


def test_r219_verified_finite_chance_reuses_only_the_matching_conditioned_cache() -> None:
    heads = _obs("heads")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached(heads, (2,), finite_outcome_id="heads"),),
        )
    )
    controller = _controller(planner, _Ticks(0.0, 1.0, 1.0, 1.0, 1.1, 1.1))
    identity = R215TurnIdentity(0, "turn")

    controller.act(identity, _obs("root"))
    reused = controller.act(
        identity,
        heads,
        boundary_reason="finite_chance",
        finite_chance_receipt=_coin_receipt(heads),
    )

    assert reused.source == "verified_finite_chance_cached_branch"
    assert len(planner.requests) == 1
    assert reused.receipt["finite_chance_cache_reused"] is True
    assert reused.receipt["finite_chance_exact_enumeration_engine_attested"] is True
    assert reused.receipt["finite_chance_receipt_validation"] == (
        "engine_receipt_and_realized_successor_validated"
    )


def test_r219_unverified_finite_chance_researches_instead_of_reusing_cache() -> None:
    heads = _obs("heads")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached(heads, (2,), finite_outcome_id="heads"),),
        ),
        R215PlanResult(selected_action=(0,), sims_run=1),
    )
    controller = _controller(planner, _Ticks(0.0, 1.0, 1.0, 1.0, 2.0, 2.0))
    identity = R215TurnIdentity(0, "turn")

    controller.act(identity, _obs("root"))
    rebuilt = controller.act(identity, heads, boundary_reason="finite_chance")

    assert rebuilt.source == "belief_mcts"
    assert len(planner.requests) == 2
    assert planner.requests[1].search_boundary_reason == "unverified_finite_chance_boundary"
    assert rebuilt.receipt["finite_chance_cache_reused"] is False
    assert rebuilt.receipt["finite_chance_rebuild_count_this_turn"] == 1


def test_r219_finite_chance_receipt_must_match_the_realized_successor() -> None:
    heads = _obs("heads")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached(heads, (2,), finite_outcome_id="heads"),),
        ),
        R215PlanResult(selected_action=(0,), sims_run=1),
    )
    controller = _controller(planner, _Ticks(0.0, 1.0, 1.0, 1.0, 2.0, 2.0))
    identity = R215TurnIdentity(0, "turn")
    bad_receipt = R219FiniteChanceReceipt(
        engine_force_enumeration_receipt_id="engine:coin:wrong-successor",
        engine_verified_exact_enumeration=True,
        outcomes=(
            R219FiniteChanceOutcome("heads", 1, 2, "public:other", "legal:other"),
            R219FiniteChanceOutcome("tails", 1, 2, "public:tails", "legal:tails"),
        ),
        realized_outcome_id="heads",
    )

    controller.act(identity, _obs("root"))
    rebuilt = controller.act(
        identity,
        heads,
        boundary_reason="finite_chance",
        finite_chance_receipt=bad_receipt,
    )

    assert rebuilt.source == "belief_mcts"
    assert len(planner.requests) == 2
    assert rebuilt.receipt["finite_chance_cache_reused"] is False
    assert rebuilt.receipt["finite_chance_exact_enumeration_engine_attested"] is False
    assert rebuilt.receipt["finite_chance_receipt_validation"] == (
        "engine_receipt_realized_successor_mismatch_research_boundary"
    )


def test_r219_reject_hook_restores_policy_before_exact_direct_fallback() -> None:
    history: list[str] = []

    class TransactionalPlanner:
        def __init__(self) -> None:
            self.requests: list[R215PlanRequest] = []
            self.accepted = 0
            self.rejected: list[str] = []
            self._history_size_before_search: int | None = None

        def plan_turn(self, request: R215PlanRequest) -> R215PlanResult:
            self.requests.append(request)
            self._history_size_before_search = len(history)
            history.append("mcts-mutated-before-validation")
            return R215PlanResult(selected_action=(1,), sims_run=0)

        def accept_plan(self, _plan: R215PlanResult) -> None:
            self.accepted += 1
            self._history_size_before_search = None

        def reject_plan(self, reason: str) -> None:
            self.rejected.append(reason)
            assert self._history_size_before_search is not None
            del history[self._history_size_before_search :]
            self._history_size_before_search = None

    planner = TransactionalPlanner()

    def direct(_observation: R215Observation) -> tuple[int, ...]:
        # This is the semantic assertion a PolicyAgent direct callback needs:
        # no speculative MCTS history survives its rejected result.
        assert history == []
        history.append("direct")
        return (0,)

    controller = R219MultiSearchTurnBeliefMCTS(
        planner,
        direct_policy=direct,
        monotonic=_Ticks(0.0, 1.0, 1.0),
    )

    decision = controller.act(R215TurnIdentity(0, "turn"), _obs("root"))

    assert decision.source == "direct_policy_fallback"
    assert decision.receipt["fallback_reason"] == "no_valid_simulation"
    assert decision.receipt["planner_result_transaction_resolution"] == "rejected"
    assert planner.accepted == 0
    assert planner.rejected == ["no_valid_simulation"]
    assert history == ["direct"]


def test_r219_late_search_rolls_back_then_charges_the_direct_fallback() -> None:
    history: list[str] = []

    class TransactionalPlanner:
        def __init__(self) -> None:
            self._history_size_before_search: int | None = None
            self.rejected: list[str] = []

        def plan_turn(self, _request: R215PlanRequest) -> R215PlanResult:
            self._history_size_before_search = len(history)
            history.append("speculative-mcts")
            return R215PlanResult(selected_action=(1,), sims_run=1)

        def accept_plan(self, _plan: R215PlanResult) -> None:
            raise AssertionError("late MCTS result must not retain authority")

        def reject_plan(self, reason: str) -> None:
            self.rejected.append(reason)
            assert self._history_size_before_search is not None
            del history[self._history_size_before_search :]
            self._history_size_before_search = None

    planner = TransactionalPlanner()

    def direct(_observation: R215Observation) -> tuple[int, ...]:
        assert history == []
        history.append("direct")
        return (0,)

    dispatched: list[list[int]] = []
    controller = R219MultiSearchTurnBeliefMCTS(
        planner,
        direct_policy=direct,
        # Search crosses the 15s segment ceiling at the pre-dispatch
        # authority checkpoint; fallback/validation/dispatch then consume 4s.
        monotonic=_Ticks(0.0, 16.0, 20.0),
    )

    decision = controller.act(
        R215TurnIdentity(0, "turn"), _obs("root"), dispatch=dispatched.append
    )

    assert decision.source == "direct_policy_fallback"
    assert decision.selected_action == (0,)
    assert planner.rejected == ["first_decision_search_budget_breach"]
    assert history == ["direct"]
    assert dispatched == [[0]]
    assert decision.receipt["pre_dispatch_controller_wall_seconds"] == 16.0
    assert decision.receipt["final_validation_and_dispatch_wall_seconds"] == 4.0
    assert decision.receipt["turn_planner_wall_seconds"] == 20.0
    assert decision.receipt["planner_seconds_used_this_turn"] == 20.0
    assert decision.receipt["planner_seconds_residual_this_turn"] == 25.0


def test_r219_final_validation_and_dispatch_are_charged_to_the_turn_pool() -> None:
    planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    dispatched: list[list[int]] = []
    controller = _controller(planner, _Ticks(0.0, 2.0, 5.0))

    decision = controller.act(
        R215TurnIdentity(0, "turn"), _obs("root"), dispatch=dispatched.append
    )

    assert decision.source == "belief_mcts"
    assert dispatched == [[1]]
    assert decision.receipt["pre_dispatch_controller_wall_seconds"] == 2.0
    assert decision.receipt["final_validation_and_dispatch_wall_seconds"] == 3.0
    assert decision.receipt["turn_planner_wall_seconds"] == 5.0
    assert decision.receipt["planner_wall_seconds_used_after"] == 5.0
    assert decision.receipt["remaining_turn_planner_pool_seconds_after"] == 40.0
    assert decision.receipt["planner_seconds_used_this_turn"] == 5.0
    assert decision.receipt["planner_seconds_residual_this_turn"] == 40.0


def test_r219_bridge_charges_raw_validation_and_external_dispatch(
    monkeypatch,
) -> None:
    observation = _obs("root")
    monkeypatch.setattr(
        r219_module,
        "r219_observation_from_raw",
        lambda _raw, _policy: observation,
    )
    planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    controller = _controller(
        planner,
        # bridge start/raw validation, controller start/pre/final,
        # bridge post-controller bookkeeping/dispatch completion
        _Ticks(0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0),
    )
    bridge = R219PolicyTurnBridge(controller, object())
    dispatched: list[list[int]] = []

    decision = bridge.act_from_raw(
        {"opaque": "raw"},
        R215TurnIdentity(0, "turn"),
        dispatch=dispatched.append,
    )

    assert decision.source == "belief_mcts"
    assert dispatched == [[1]]
    assert decision.receipt["bridge_public_observation_validation_wall_seconds"] == 1.0
    assert (
        decision.receipt[
            "bridge_post_selection_commit_and_dispatch_wall_seconds"
        ]
        == 2.0
    )
    assert decision.receipt["bridge_validation_and_dispatch_wall_seconds"] == 3.0
    assert decision.receipt["turn_planner_wall_seconds"] == 5.0
    assert decision.receipt["planner_seconds_used_this_turn"] == 5.0
    assert decision.receipt["planner_seconds_residual_this_turn"] == 40.0
    assert decision.receipt["real_actions_dispatched"] == 1


def test_r219_turn_close_discards_pool_and_reports_canary_metrics() -> None:
    planner = _Planner(
        R215PlanResult(selected_action=(1,), sims_run=1),
        R215PlanResult(selected_action=(2,), sims_run=1),
    )
    controller = _controller(planner, _Ticks(0.0, 2.0, 2.0, 2.0, 5.0, 5.0))
    identity = R215TurnIdentity(0, "turn")

    controller.act(identity, _obs("root"))
    close = controller.finish_actual_turn()
    next_turn = controller.act(identity, _obs("fresh-real-turn"))

    assert close is not None
    assert close["schema"].endswith("r219/v1")
    assert close["actual_turn_closed"] is True
    assert close["search_segments_this_turn"] == 1
    assert close["only_first_segment_search"] is True
    assert close["planner_seconds_used_this_turn"] == 2.0
    assert close["planner_seconds_residual_this_turn"] == 43.0
    assert next_turn.source == "belief_mcts"
    assert len(planner.requests) == 2


def test_r219_native_result_helper_requires_no_sampled_chance_cache_attestation() -> None:
    safe_diagnostics = {
        "principal_continuation_by_fingerprint": [
            {
                "expected_public_observation_fingerprint": "public:next",
                "legal_actions": [[0], [1]],
                "selected_action": [1],
                "deterministic": True,
                "selected_action_fully_backed_up": True,
                "no_sampled_or_opaque_chance": True,
            }
        ]
    }
    unsafe_diagnostics = {
        "principal_continuation_by_fingerprint": [
            {
                "expected_public_observation_fingerprint": "public:next",
                "legal_actions": [[0], [1]],
                "selected_action": [1],
                "deterministic": True,
                "selected_action_fully_backed_up": True,
            }
        ]
    }
    safe = r219_plan_result_from_mcts_result(
        SimpleNamespace(
            select=[0],
            sims_run=1,
            target=SimpleNamespace(diagnostics=safe_diagnostics),
        )
    )
    unsafe = r219_plan_result_from_mcts_result(
        SimpleNamespace(
            select=[0],
            sims_run=1,
            target=SimpleNamespace(diagnostics=unsafe_diagnostics),
        )
    )

    assert len(safe.continuation) == 1
    assert unsafe.continuation == ()


def test_adapter_uses_root_action_stable_key_only_with_a_complete_receipt() -> None:
    receipt = {
        "stable_root_convergence": True,
        "selected_action": [1],
        "selected_action_legal": True,
        "selected_action_fully_backed_up": True,
        "selected_action_visit_count": 2,
        "completed_backups": 2,
        "root_value_or_visit_stability_evidence": {"visit_share": 1.0},
    }

    class Engine:
        min_trusted_sims = 1
        max_depth = 5

        def search(self, *_args, **_kwargs):
            return SimpleNamespace(
                select=[1],
                sims_run=2,
                target=SimpleNamespace(
                    diagnostics={
                        "root_action_stable": True,
                        "root_stability_receipt": receipt,
                    }
                ),
            )

    adapter = BeliefMCTSPlannerAdapter(
        Engine(),
        belief_history_for=lambda _request: object(),
        root_history_for=lambda _request: ((), ()),
    )
    request = R215PlanRequest(
        turn_identity=R215TurnIdentity(0, "turn"),
        observation=R215Observation("public:root", ((0,), (1,)), raw_observation={}),
        turn_pool_remaining_seconds=45.0,
        operation_allowance_seconds=15.0,
        first_decision_search_allowance_seconds=15.0,
        game_clock_remaining_seconds=600.0,
        timing_identity_sha256="sha256:test",
        emergency_simulation_safety_ceiling=1_000_000,
        emergency_depth_safety_ceiling=1_000_000,
    )

    result = adapter.plan_turn(request)

    assert result.root_action_stable is True
    assert result.root_stability_receipt == receipt
