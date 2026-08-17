"""Swap-in wiring tests for Recursive Turn Planner ↔ PolicyAgent."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from poke_bot import features
from poke_bot.agent import PolicyAgent
from poke_bot.recursive_turn_planner import (
    NodeKind,
    PlanExecutor,
    PlanNode,
    RecursiveTurnPlanner,
    RTPConfig,
    TurnDecision,
    TurnProgram,
    resolve_rtp_config_for_model,
    turn_key_from_obs,
)
from poke_bot.recursive_turn_planner.agent_bridge import RTPAgentBridge


def _mock_model(d_model: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        d_model=d_model,
        latent_lookahead=None,
        latent_lookahead_enabled=False,
        eval=lambda: None,
        parameters=lambda: iter((torch.zeros(1),)),
    )


def _cursor_program(
    *actions: tuple[int, ...],
    plan_id: str,
) -> TurnProgram:
    nodes = tuple(
        PlanNode(kind=NodeKind.PRIMITIVE, action=tuple(action))
        for action in actions
    )
    assert nodes
    root = (
        nodes[0]
        if len(nodes) == 1
        else PlanNode(kind=NodeKind.SEQUENCE, children=nodes)
    )
    return TurnProgram(root=root, plan_id=plan_id)


def _scripted_cursor_bridge(
    *,
    legal_by_select: list[tuple[tuple[int, ...], ...]],
    decisions: list[TurnDecision],
    repair_budget: int = 0,
) -> tuple[RTPAgentBridge, list[bool | None]]:
    """Build a bridge whose encoder and planner are deterministic test seams."""

    config = RTPConfig(
        sizing_profile="cursor_test",
        d_model=8,
        dynamics_width=16,
        num_plan_candidates=1,
        max_recursion_depth=1,
        max_neural_passes=8,
        repair_budget=repair_budget,
    )
    planner = RecursiveTurnPlanner(config)
    bridge = RTPAgentBridge(
        model=_mock_model(8),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=config,
        planner=planner,
        get_matchup_route=lambda: -1,
        get_board_history=lambda: [],
        get_previous_action_history=lambda: [],
        get_previous_action_token=lambda: None,
        get_kv_cache=lambda: None,
        set_kv_cache=lambda _cache: None,
    )
    scripted_legal = iter(legal_by_select)
    scripted_decisions = iter(decisions)
    force_recurse_calls: list[bool | None] = []

    def fake_legal(_obs_dict: dict) -> tuple[tuple[int, ...], ...]:
        return next(scripted_legal)

    def fake_encode(
        _obs_dict: dict,
        *,
        board: features.SparseVector,
        legal_actions: tuple[tuple[int, ...], ...],
        append_cache: bool = True,
    ) -> tuple[object, torch.Tensor]:
        _ = board, append_cache
        memory = planner.encode_memory(
            torch.zeros(config.d_model),
            legal_actions=legal_actions,
        )
        return memory, torch.zeros(len(legal_actions))

    def fake_plan_turn(
        _memory: object,
        *,
        policy_logits: torch.Tensor | None = None,
        force_recurse: bool | None = None,
    ) -> TurnDecision:
        _ = policy_logits
        force_recurse_calls.append(force_recurse)
        return next(scripted_decisions)

    bridge._legal_actions = fake_legal  # type: ignore[method-assign]
    bridge.encode = fake_encode  # type: ignore[method-assign]
    planner.plan_turn = fake_plan_turn  # type: ignore[method-assign]
    return bridge, force_recurse_calls


def _same_turn_obs() -> dict[str, dict[str, int]]:
    return {"current": {"yourIndex": 0, "turn": 7}}


@pytest.mark.unit
def test_turn_key_from_obs_dict() -> None:
    assert turn_key_from_obs({"current": {"yourIndex": 1, "turn": 7}}) == (1, 7)


@pytest.mark.unit
def test_resolve_rtp_config_binds_to_model_width() -> None:
    cfg = resolve_rtp_config_for_model(_mock_model(16))  # type: ignore[arg-type]
    assert cfg.d_model == 16
    assert cfg.dynamics_width == 32
    pure = resolve_rtp_config_for_model(None, profile_name="pure_rl")
    assert pure.d_model == 96
    r197 = resolve_rtp_config_for_model(None, profile_name="pure_rl_r197")
    assert r197.sizing_profile == "pure_rl_r197"
    assert r197.max_neural_passes == 256
    global_cfg = resolve_rtp_config_for_model(_mock_model(256))  # type: ignore[arg-type]
    assert global_cfg.sizing_profile == "global_transformer"
    assert global_cfg.dynamics_width == 512


@pytest.mark.unit
def test_policy_agent_inits_rtp_bridge_when_explicitly_enabled() -> None:
    model = _mock_model(16)
    # PolicyAgent.__post_init__ calls model.eval() and may touch parameters().
    agent = PolicyAgent(
        model=model,  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        use_recursive_turn_planner=True,
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent.use_recursive_turn_planner is True
    assert agent._rtp_bridge is not None
    assert isinstance(agent._rtp_bridge, RTPAgentBridge)
    assert agent._rtp_bridge.config.d_model == 16
    assert agent._rtp_bridge.max_action_combos == 256
    agent._rtp_bridge.active_turn_key = (0, 3)
    agent.reset_game()
    assert agent._rtp_bridge.active_turn_key == (-1, -1)
    assert agent._rtp_bridge.memory is None


@pytest.mark.unit
def test_policy_agent_can_disable_rtp() -> None:
    agent = PolicyAgent(
        model=_mock_model(),  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        use_recursive_turn_planner=False,
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent._rtp_bridge is None


@pytest.mark.unit
def test_policy_agent_without_model_skips_bridge() -> None:
    agent = PolicyAgent(
        model=None,
        deck=[1] * 60,
        use_mcts=False,
        matchup_adapter_shadow=False,
    )
    assert agent._rtp_bridge is None


@pytest.mark.unit
def test_policy_agent_r197_uses_exact_complete_action_cap() -> None:
    agent = PolicyAgent(
        model=_mock_model(96),  # type: ignore[arg-type]
        deck=[1] * 60,
        use_mcts=False,
        use_recursive_turn_planner=True,
        rtp_sizing_profile="pure_rl_r197",
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent._rtp_bridge is not None
    assert agent._rtp_bridge.config.sizing_profile == "pure_rl_r197"
    assert agent._rtp_bridge.max_action_combos == 1024

    with pytest.raises(ValueError, match="requires rtp_max_action_combos=1024"):
        PolicyAgent(
            model=_mock_model(96),  # type: ignore[arg-type]
            deck=[1] * 60,
            use_mcts=False,
            use_recursive_turn_planner=True,
            rtp_sizing_profile="pure_rl_r197",
            rtp_max_action_combos=256,
            matchup_adapter_shadow=False,
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_action_space_too_large_special_is_scoped_to_r197_1024(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only r198's exact bridge profile can emit the zero-pass special trace."""

    def callbacks() -> dict[str, object]:
        return {
            "get_matchup_route": lambda: -1,
            "get_board_history": lambda: [],
            "get_previous_action_history": lambda: [],
            "get_previous_action_token": lambda: None,
            "get_kv_cache": lambda: None,
            "set_kv_cache": lambda _cache: None,
        }

    observation = {
        "current": {"yourIndex": 0, "turn": 7},
        "select": {
            "minCount": 1,
            "maxCount": 5,
            "option": [{"type": 2} for _ in range(9)],
        },
    }
    board = features.SparseVector()

    def too_large(_observation: dict) -> tuple[tuple[int, ...], ...]:
        raise features.ActionSpaceTooLarge("18,729 complete ordered actions")

    r197_config = resolve_rtp_config_for_model(
        _mock_model(96),  # type: ignore[arg-type]
        profile_name="pure_rl_r197",
    )
    r197 = RTPAgentBridge(
        model=_mock_model(96),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=r197_config,
        planner=RecursiveTurnPlanner(r197_config),
        **callbacks(),  # type: ignore[arg-type]
    )
    # This unit exercise is about the bridge's post-enumeration catch, not
    # the separately sealed evaluation-action fence.
    monkeypatch.setattr(r197, "_require_r197_action_selection_authority", lambda: None)
    r197._legal_actions = too_large  # type: ignore[method-assign]
    assert r197.select(
        observation, board=board, greedy_fallback=lambda _obs: [0, 2, 3, 6, 5]
    ) == [0, 2, 3, 6, 5]
    r197_diag = r197.last_diagnostics.as_dict()
    assert r197_diag["mode"] == "fallback"
    assert r197_diag["fallback_code"] == "action_space_too_large"
    assert r197_diag["neural_passes"] == 0
    assert r197_diag["required_neural_passes"] == 0
    assert r197_diag["legal_count"] == 0
    assert r197_diag["decision_mode"] == ""
    assert r197_diag["extras"]["over_cap_factorized_fallback"] == {
        "classification": "complete_ordered_action_space_over_cap",
        "action_space": features.complete_ordered_action_space_summary(
            observation, max_combos=1024
        ),
        "factorized_greedy_fallback": True,
    }

    legacy_config = RTPConfig(
        sizing_profile="cursor_test",
        d_model=8,
        dynamics_width=16,
        num_plan_candidates=1,
        max_recursion_depth=1,
        max_neural_passes=8,
    )
    legacy = RTPAgentBridge(
        model=_mock_model(8),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=legacy_config,
        planner=RecursiveTurnPlanner(legacy_config),
        max_action_combos=1024,
        **callbacks(),  # type: ignore[arg-type]
    )
    legacy._legal_actions = too_large  # type: ignore[method-assign]
    assert legacy.select(observation, board=board, greedy_fallback=lambda _obs: [0]) == [0]
    legacy_diag = legacy.last_diagnostics.as_dict()
    assert legacy_diag["fallback_code"] == "action_space_too_large"
    assert legacy_diag["required_neural_passes"] != 0
    assert "over_cap_factorized_fallback" not in legacy_diag["extras"]


@pytest.mark.unit
def test_bridge_uses_planner_config_for_executor_and_diagnostics() -> None:
    planner_config = resolve_rtp_config_for_model(
        _mock_model(96),  # type: ignore[arg-type]
        profile_name="pure_rl_r197",
    )
    planner = RecursiveTurnPlanner(planner_config)
    callbacks = {
        "get_matchup_route": lambda: -1,
        "get_board_history": lambda: [],
        "get_previous_action_history": lambda: [],
        "get_previous_action_token": lambda: None,
        "get_kv_cache": lambda: None,
        "set_kv_cache": lambda _cache: None,
    }
    bridge = RTPAgentBridge(
        model=_mock_model(96),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=planner_config,
        planner=planner,
        **callbacks,
    )
    assert bridge.config == planner_config
    assert bridge.executor is not None
    assert bridge.executor.config == planner_config

    diag = bridge._new_diagnostics((0, 1))
    bridge._record_fallback(
        diag, code="neural_pass_budget_exceeded", detail="257 > 256"
    )
    record = diag.as_dict()
    assert record["required_neural_passes"] == 6
    assert record["fallback_code"] == "neural_pass_budget_exceeded"
    assert record["planner_config"]["max_neural_passes"] == 256
    assert record["planner_config"]["max_action_combos"] == 1024

    with pytest.raises(ValueError, match="requires max_action_combos=1024"):
        RTPAgentBridge(
            model=_mock_model(96),  # type: ignore[arg-type]
            deck=[1] * 60,
            config=planner_config,
            planner=planner,
            max_action_combos=256,
            **callbacks,
        )

    with pytest.raises(ValueError, match="promotion receipt"):
        RTPAgentBridge(
            model=_mock_model(96),  # type: ignore[arg-type]
            deck=[1] * 60,
            config=planner_config,
            serving_qualified=True,
            expected_parent_digest="sha256:" + "a" * 64,
            planner=planner,
            **callbacks,
        )

    with pytest.raises(ValueError, match="must exactly match"):
        RTPAgentBridge(
            model=_mock_model(96),  # type: ignore[arg-type]
            deck=[1] * 60,
            config=resolve_rtp_config_for_model(None, profile_name="pure_rl"),
            planner=planner,
            **callbacks,
        )

    with pytest.raises(ValueError, match="executor config does not match"):
        RTPAgentBridge(
            model=_mock_model(96),  # type: ignore[arg-type]
            deck=[1] * 60,
            config=planner_config,
            planner=planner,
            executor=PlanExecutor(
                resolve_rtp_config_for_model(
                    None,
                    profile_name="pure_rl_r197",
                    online_sim_verify_budget=1,
                )
            ),
            **callbacks,
        )


@pytest.mark.unit
def test_r197_executor_repair_is_replanned_through_visible_bridge_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid persisted program cannot hide a forced replan's five passes."""

    planner_config = resolve_rtp_config_for_model(
        _mock_model(96),  # type: ignore[arg-type]
        profile_name="pure_rl_r197",
    )
    planner = RecursiveTurnPlanner(planner_config)
    callbacks = {
        "get_matchup_route": lambda: -1,
        "get_board_history": lambda: [],
        "get_previous_action_history": lambda: [],
        "get_previous_action_token": lambda: None,
        "get_kv_cache": lambda: None,
        "set_kv_cache": lambda _cache: None,
    }
    bridge = RTPAgentBridge(
        model=_mock_model(96),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=planner_config,
        planner=planner,
        **callbacks,
    )
    # This focused test is about the executor/bridge handoff rather than the
    # separately covered sealed evaluator-authority gate.
    monkeypatch.setattr(
        bridge,
        "_require_r197_action_selection_authority",
        lambda: {"evaluation_only": True},
    )
    memory = planner.encode_memory(
        torch.zeros(96),
        legal_actions=((0,), (1,)),
    )
    assert bridge.executor is not None
    bridge.memory = memory
    bridge.active_turn_key = (0, 7)
    bridge.executor.load(
        TurnProgram(
            root=PlanNode(kind=NodeKind.PRIMITIVE, action=(99,)),
            plan_id="stale",
        )
    )
    monkeypatch.setattr(bridge, "_legal_actions", lambda _obs: ((0,), (1,)))
    monkeypatch.setattr(
        bridge,
        "encode",
        lambda *_args, **_kwargs: (memory, torch.zeros(2)),
    )

    action = bridge.select(
        {"current": {"yourIndex": 0, "turn": 7}},
        board=object(),  # type: ignore[arg-type]
        greedy_fallback=lambda _obs: [0],
    )

    record = bridge.last_diagnostics.as_dict()
    assert action in ([0], [1])
    assert record["mode"] == "replan_with_program"
    assert record["required_neural_passes"] == 5
    assert record["neural_passes"] == 5
    assert record["decision_mode"] == "recursive_plan"


@pytest.mark.unit
def test_r197_changed_legal_nested_cursor_replans_and_consumes_new_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An illegal nested continuation reaches an explicit five-pass replan."""

    torch.manual_seed(198)
    planner_config = resolve_rtp_config_for_model(
        _mock_model(96),  # type: ignore[arg-type]
        profile_name="pure_rl_r197",
    )
    planner = RecursiveTurnPlanner(planner_config)
    bridge = RTPAgentBridge(
        model=_mock_model(96),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=planner_config,
        planner=planner,
        get_matchup_route=lambda: -1,
        get_board_history=lambda: [],
        get_previous_action_history=lambda: [],
        get_previous_action_token=lambda: None,
        get_kv_cache=lambda: None,
        set_kv_cache=lambda _cache: None,
    )
    monkeypatch.setattr(
        bridge,
        "_require_r197_action_selection_authority",
        lambda: {"evaluation_only": True},
    )

    current_legal = [[(index,) for index in range(8)]]

    def fake_legal(_obs: dict) -> tuple[tuple[int, ...], ...]:
        return tuple(current_legal[0])

    def fake_encode(
        _obs: dict,
        *,
        board: features.SparseVector,
        legal_actions: tuple[tuple[int, ...], ...],
        append_cache: bool = True,
    ) -> tuple[object, torch.Tensor]:
        _ = board, append_cache
        return (
            planner.encode_memory(
                torch.zeros(96),
                legal_actions=legal_actions,
            ),
            torch.zeros(len(legal_actions)),
        )

    monkeypatch.setattr(bridge, "_legal_actions", fake_legal)
    monkeypatch.setattr(bridge, "encode", fake_encode)
    obs = _same_turn_obs()
    board = features.SparseVector()
    greedy = lambda _obs: [999]

    first_action = bridge.select(obs, board=board, greedy_fallback=greedy)
    first_record = bridge.last_diagnostics.as_dict()
    assert tuple(first_action) in current_legal[0]
    assert first_record["mode"] == "recursive_plan"
    assert first_record["required_neural_passes"] == 6
    assert first_record["neural_passes"] == 6
    assert first_record["fallback_code"] == ""
    assert bridge.executor is not None
    assert bridge.executor.repair_fn is None
    initial_cursor = bridge.executor.cursor
    assert initial_cursor is not None
    assert initial_cursor.kind is NodeKind.SEQUENCE
    assert initial_cursor.children[0].kind is NodeKind.SEQUENCE
    assert bridge.executor.steps_executed == 1

    # Every nested action in the persisted plan belongs to the old legal set.
    # A new same-turn prompt therefore exercises extracted_action_illegal,
    # which must be an unrepaired abort followed by a visible forced replan.
    current_legal[0] = [(100 + index,) for index in range(8)]
    second_action = bridge.select(obs, board=board, greedy_fallback=greedy)
    second_record = bridge.last_diagnostics.as_dict()
    assert tuple(second_action) in current_legal[0]
    assert second_record["mode"] == "replan_with_program"
    assert second_record["required_neural_passes"] == 5
    assert second_record["neural_passes"] == 5
    assert second_record["fallback_code"] == ""
    loaded = second_record["extras"]["loaded_program_first_step"]
    assert loaded["phase"] == "replan"
    assert loaded["expected_action"] == second_action
    assert loaded["executor_action"] == second_action
    assert loaded["repaired"] is False
    assert bridge.executor.steps_executed == 1
    assert bridge.executor.active_program is not None
    assert bridge.executor.cursor is not None
    assert bridge.executor.cursor != bridge.executor.active_program.root

    expected_continuation = bridge.executor.cursor.first_action()
    assert expected_continuation is not None
    third_action = bridge.select(obs, board=board, greedy_fallback=greedy)
    third_record = bridge.last_diagnostics.as_dict()
    assert third_action == list(expected_continuation)
    assert third_record["mode"] == "continue_plan"
    assert third_record["neural_passes"] == 0
    assert third_record["fallback_code"] == ""
    assert bridge.executor.steps_executed == 2


@pytest.mark.unit
def test_r197_rejects_executor_owned_repair_callback() -> None:
    """A custom executor callback could otherwise spend unreported passes."""

    planner_config = resolve_rtp_config_for_model(
        _mock_model(96),  # type: ignore[arg-type]
        profile_name="pure_rl_r197",
    )
    planner = RecursiveTurnPlanner(planner_config)
    callbacks = {
        "get_matchup_route": lambda: -1,
        "get_board_history": lambda: [],
        "get_previous_action_history": lambda: [],
        "get_previous_action_token": lambda: None,
        "get_kv_cache": lambda: None,
        "set_kv_cache": lambda _cache: None,
    }
    with pytest.raises(ValueError, match="must not use an executor repair_fn"):
        RTPAgentBridge(
            model=_mock_model(96),  # type: ignore[arg-type]
            deck=[1] * 60,
            config=planner_config,
            planner=planner,
            executor=PlanExecutor(
                planner_config,
                repair_fn=lambda _memory, program: program,
            ),
            **callbacks,
        )


@pytest.mark.unit
def test_bridge_consumes_initial_program_action_before_same_turn_continue() -> None:
    program = _cursor_program((1,), (2,), plan_id="initial")
    bridge, plan_calls = _scripted_cursor_bridge(
        legal_by_select=[((1,), (2,)), ((2,),)],
        decisions=[
            TurnDecision(
                mode="recursive_plan",
                action=(1,),
                program=program,
                neural_passes=2,
            )
        ],
    )
    board = features.SparseVector()
    greedy = lambda _obs: [99]

    assert bridge.select(_same_turn_obs(), board=board, greedy_fallback=greedy) == [1]
    assert bridge.executor is not None
    assert bridge.executor.steps_executed == 1
    assert bridge.last_diagnostics.extras["loaded_program_first_step"] == {
        "phase": "initial",
        "expected_action": [1],
        "executor_action": [1],
        "done": False,
        "repaired": False,
        "reason": "ok",
    }

    assert bridge.select(_same_turn_obs(), board=board, greedy_fallback=greedy) == [2]
    assert bridge.last_diagnostics.mode == "continue_plan"
    assert bridge.executor.steps_executed == 2
    assert plan_calls == [None]


@pytest.mark.unit
def test_bridge_accepts_legal_empty_initial_program_root() -> None:
    """A zero-selection prompt may legitimately execute an empty root action."""

    program = _cursor_program((), plan_id="legal-empty-initial")
    bridge, plan_calls = _scripted_cursor_bridge(
        legal_by_select=[((),)],
        decisions=[
            TurnDecision(
                mode="recursive_plan",
                action=(),
                program=program,
                neural_passes=2,
            )
        ],
    )

    fallback_calls: list[dict] = []
    action = bridge.select(
        _same_turn_obs(),
        board=features.SparseVector(),
        greedy_fallback=lambda obs: fallback_calls.append(obs) or [99],
    )

    assert action == []
    assert fallback_calls == []
    record = bridge.last_diagnostics.as_dict()
    assert record["mode"] == "recursive_plan"
    assert record["fallback_code"] == ""
    assert record["extras"]["loaded_program_first_step"] == {
        "phase": "initial",
        "expected_action": [],
        "executor_action": [],
        "done": False,
        "repaired": False,
        "reason": "ok",
    }


@pytest.mark.unit
def test_bridge_consumes_replan_program_action_before_same_turn_continue() -> None:
    initial = _cursor_program((1,), (2,), plan_id="initial")
    replanned = _cursor_program((3,), (4,), plan_id="replanned")
    bridge, plan_calls = _scripted_cursor_bridge(
        legal_by_select=[((1,), (2,)), ((3,), (4,)), ((4,),)],
        decisions=[
            TurnDecision(
                mode="recursive_plan",
                action=(1,),
                program=initial,
                neural_passes=2,
            ),
            TurnDecision(
                mode="recursive_plan",
                action=(3,),
                program=replanned,
                neural_passes=1,
            ),
        ],
    )
    board = features.SparseVector()
    greedy = lambda _obs: [99]

    assert bridge.select(_same_turn_obs(), board=board, greedy_fallback=greedy) == [1]
    # The old remaining action (2,) is no longer legal. With repair disabled,
    # this forces the bridge's replan-with-program path.
    assert bridge.select(_same_turn_obs(), board=board, greedy_fallback=greedy) == [3]
    assert bridge.last_diagnostics.mode == "replan_with_program"
    assert bridge.last_diagnostics.extras["loaded_program_first_step"]["phase"] == "replan"
    assert bridge.executor is not None
    assert bridge.executor.steps_executed == 1

    # The replan root was already consumed, so the next select advances to 4.
    assert bridge.select(_same_turn_obs(), board=board, greedy_fallback=greedy) == [4]
    assert bridge.last_diagnostics.mode == "continue_plan"
    assert bridge.executor.steps_executed == 2
    assert plan_calls == [None, True]


@pytest.mark.unit
def test_r197_replan_accepts_legal_empty_program_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible five-pass replan may correctly choose the empty legal action."""

    config = resolve_rtp_config_for_model(
        _mock_model(96),  # type: ignore[arg-type]
        profile_name="pure_rl_r197",
    )
    planner = RecursiveTurnPlanner(config)
    bridge = RTPAgentBridge(
        model=_mock_model(96),  # type: ignore[arg-type]
        deck=[1] * 60,
        config=config,
        planner=planner,
        get_matchup_route=lambda: -1,
        get_board_history=lambda: [],
        get_previous_action_history=lambda: [],
        get_previous_action_token=lambda: None,
        get_kv_cache=lambda: None,
        set_kv_cache=lambda _cache: None,
    )
    monkeypatch.setattr(
        bridge,
        "_require_r197_action_selection_authority",
        lambda: {"evaluation_only": True},
    )
    stale_memory = planner.encode_memory(
        torch.zeros(96), legal_actions=((0,),)
    )
    current_memory = planner.encode_memory(
        torch.zeros(96), legal_actions=((), (0,))
    )
    assert bridge.executor is not None
    assert bridge.executor.repair_fn is None
    bridge.memory = stale_memory
    bridge.active_turn_key = (0, 7)
    bridge.executor.load(_cursor_program((99,), plan_id="stale"))
    monkeypatch.setattr(bridge, "_legal_actions", lambda _obs: ((), (0,)))
    monkeypatch.setattr(
        bridge,
        "encode",
        lambda *_args, **_kwargs: (current_memory, torch.zeros(2)),
    )
    force_recurse_calls: list[bool | None] = []

    def fake_plan_turn(
        _memory: object,
        *,
        policy_logits: torch.Tensor | None = None,
        force_recurse: bool | None = None,
    ) -> TurnDecision:
        _ = policy_logits
        force_recurse_calls.append(force_recurse)
        assert force_recurse is True
        planner._neural_passes = 5
        return TurnDecision(
            mode="recursive_plan",
            action=(),
            program=_cursor_program((), plan_id="legal-empty-replan"),
            neural_passes=5,
            diagnostics={"complexity_gate": {"forced": True}},
        )

    monkeypatch.setattr(planner, "plan_turn", fake_plan_turn)
    fallback_calls: list[dict] = []
    action = bridge.select(
        _same_turn_obs(),
        board=features.SparseVector(),
        greedy_fallback=lambda obs: fallback_calls.append(obs) or [99],
    )

    assert action == []
    assert fallback_calls == []
    assert force_recurse_calls == [True]
    record = bridge.last_diagnostics.as_dict()
    assert record["mode"] == "replan_with_program"
    assert record["required_neural_passes"] == 5
    assert record["neural_passes"] == 5
    assert record["fallback_code"] == ""
    assert record["extras"]["loaded_program_first_step"] == {
        "phase": "replan",
        "expected_action": [],
        "executor_action": [],
        "done": False,
        "repaired": False,
        "reason": "ok",
    }


@pytest.mark.unit
@pytest.mark.parametrize("planned_action", [None, (99,)])
def test_bridge_falls_back_for_missing_or_nonmember_action(
    planned_action: tuple[int, ...] | None,
) -> None:
    """Only None or an action outside the complete legal set may fall back."""

    bridge, plan_calls = _scripted_cursor_bridge(
        legal_by_select=[((1,),)],
        decisions=[
            TurnDecision(
                mode="direct_policy",
                action=planned_action,  # type: ignore[arg-type]
                neural_passes=0,
                used_direct_policy=True,
            )
        ],
    )

    assert bridge.select(
        _same_turn_obs(),
        board=features.SparseVector(),
        greedy_fallback=lambda _obs: [99],
    ) == [99]
    assert bridge.last_diagnostics.mode == "fallback"
    assert bridge.last_diagnostics.fallback_code == "planned_action_not_legal"
    assert plan_calls == [None]


@pytest.mark.unit
def test_bridge_fails_closed_when_executor_first_action_disagrees() -> None:
    program = _cursor_program((1,), (2,), plan_id="mismatch")
    bridge, plan_calls = _scripted_cursor_bridge(
        legal_by_select=[((1,), (2,))],
        decisions=[
            TurnDecision(
                mode="recursive_plan",
                action=(2,),
                program=program,
                neural_passes=2,
            )
        ],
    )
    board = features.SparseVector()

    assert bridge.select(
        _same_turn_obs(), board=board, greedy_fallback=lambda _obs: [99]
    ) == [99]
    assert bridge.last_diagnostics.mode == "fallback"
    assert bridge.last_diagnostics.fallback_code == "executor_first_action_mismatch"
    assert bridge.last_diagnostics.extras["loaded_program_first_step"]["executor_action"] == [1]
    assert bridge.executor is not None
    assert bridge.executor.active_program is None
    assert plan_calls == [None]


@pytest.mark.unit
def test_bridge_fails_closed_when_load_time_executor_repairs_program() -> None:
    invalid = _cursor_program((9,), plan_id="invalid")
    repaired = _cursor_program((1,), plan_id="repaired")
    bridge, plan_calls = _scripted_cursor_bridge(
        legal_by_select=[((1,),)],
        decisions=[
            TurnDecision(
                mode="recursive_plan",
                action=(1,),
                program=invalid,
                neural_passes=2,
            ),
            TurnDecision(
                mode="recursive_plan",
                action=(1,),
                program=repaired,
                neural_passes=1,
            ),
        ],
        repair_budget=1,
    )

    assert bridge.select(
        _same_turn_obs(),
        board=features.SparseVector(),
        greedy_fallback=lambda _obs: [99],
    ) == [99]
    assert bridge.last_diagnostics.fallback_code == "executor_first_action_mismatch"
    assert bridge.last_diagnostics.extras["loaded_program_first_step"]["repaired"] is True
    assert bridge.executor is not None
    assert bridge.executor.active_program is None
    # The executor's repair passed through the normal callback, but its action
    # was never accepted without bridge-visible replan telemetry.
    assert plan_calls == [None, True]
