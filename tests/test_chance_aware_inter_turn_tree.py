from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path

import pytest

from poke_bot.recursive_turn_planner.chance_aware_tree import (
    ActionEdge,
    BranchOutcome,
    CachedActionTree,
    ChanceAwareSearchConfig,
    ChanceAwareTreeController,
    ChanceAwareTreeError,
    ChanceOutcome,
    ControllerProtocolError,
    ControllerState,
    DecisionNode,
    DecisionSnapshot,
    DeterministicTransition,
    FiniteChanceNode,
    FiniteChanceTransition,
    ObservedPublicBranchEvidence,
    ObservedPublicBranchTransition,
    PublicStateKey,
    RebuildBoundaryTransition,
    TerminalNode,
    TerminalTransition,
    complete_action_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "poke_bot/recursive_turn_planner/chance_aware_tree.py"
)


class _ScriptedClock:
    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def __call__(self) -> float:
        if not self._readings:
            raise AssertionError("scripted monotonic clock was exhausted")
        return self._readings.pop(0)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _key(
    label: str,
    *,
    turn_key: tuple[int, int],
    serial: int,
    legal_actions: tuple[tuple[int, ...], ...],
) -> PublicStateKey:
    return PublicStateKey(
        turn_key=turn_key,
        decision_serial=serial,
        observation_sha256=_digest(f"obs:{label}"),
        legal_actions_sha256=complete_action_fingerprint(legal_actions),
        option_encoding_sha256=_digest(f"options:{label}"),
        public_history_sha256=_digest(f"history:{label}"),
        source_sha256=_digest("source"),
        model_sha256=_digest("model"),
        rules_abi_sha256=_digest("rules"),
        profile_sha256=_digest("profile"),
    )


def _snapshot(
    label: str,
    *,
    turn_key: tuple[int, int],
    serial: int,
    legal_actions: tuple[tuple[int, ...], ...],
    direct_action: tuple[int, ...],
) -> DecisionSnapshot:
    return DecisionSnapshot(
        key=_key(
            label,
            turn_key=turn_key,
            serial=serial,
            legal_actions=legal_actions,
        ),
        legal_actions=legal_actions,
        direct_action=direct_action,
    )


def _decision(
    snapshot: DecisionSnapshot,
    *,
    recommended: tuple[int, ...],
    transitions: dict[tuple[int, ...], object],
    value: Fraction = Fraction(0),
) -> DecisionNode:
    edges = tuple(
        ActionEdge(action=action, transition=transitions[action])
        for action in snapshot.legal_actions
        if action in transitions
    )
    return DecisionNode(
        expected_state=snapshot.key,
        expected_legal_actions=snapshot.legal_actions,
        direct_action=snapshot.direct_action,
        shadow_recommended_action=recommended,
        search_value=value,
        edges=edges,
    )


def _tree(root: DecisionNode, config: ChanceAwareSearchConfig) -> CachedActionTree:
    return CachedActionTree(
        root=root,
        config_sha256=config.identity_sha256,
        planner_artifact_sha256=_digest("offline-phase1-fixture"),
    )


def _two_step_cross_turn_tree(
    config: ChanceAwareSearchConfig | None = None,
) -> tuple[
    ChanceAwareSearchConfig,
    DecisionSnapshot,
    DecisionSnapshot,
    CachedActionTree,
]:
    config = config or ChanceAwareSearchConfig()
    first = _snapshot(
        "first",
        turn_key=(0, 7),
        serial=10,
        legal_actions=((0,), (1,)),
        direct_action=(0,),
    )
    second = _snapshot(
        "second",
        turn_key=(0, 8),
        serial=11,
        legal_actions=((), (2,)),
        direct_action=(),
    )
    terminal = TerminalNode(Fraction(1), reason="game_terminal")
    child = _decision(
        second,
        recommended=(2,),
        transitions={
            (): TerminalTransition(terminal),
            (2,): TerminalTransition(terminal),
        },
        value=Fraction(3, 4),
    )
    root = _decision(
        first,
        recommended=(1,),
        transitions={
            (0,): DeterministicTransition(_digest("deterministic"), child),
            (1,): RebuildBoundaryTransition("shadow_alternative_unexpanded"),
        },
        value=Fraction(1, 2),
    )
    return config, first, second, _tree(root, config)


def test_budget_defaults_are_easy_to_change_and_identity_bound() -> None:
    defaults = ChanceAwareSearchConfig()
    changed = ChanceAwareSearchConfig(max_turn_seconds=30.0, max_action_seconds=3.0)

    assert defaults.max_turn_seconds == 20.0
    assert defaults.max_action_seconds == 5.0
    assert defaults.max_complete_actions == 1024
    assert changed.max_turn_seconds == 30.0
    assert changed.max_action_seconds == 3.0
    assert changed.identity_sha256 != defaults.identity_sha256
    with pytest.raises(ChanceAwareTreeError, match="cannot exceed"):
        ChanceAwareSearchConfig(max_turn_seconds=4.0, max_action_seconds=5.0)
    with pytest.raises(ChanceAwareTreeError, match=r"\[1, 1024\]"):
        ChanceAwareSearchConfig(max_complete_actions=1025)


def test_multi_action_cross_turn_tree_reuses_deterministic_child_without_rebuild() -> None:
    config, first, second, tree = _two_step_cross_turn_tree()
    controller = ChanceAwareTreeController(config)

    assert controller.install_tree(tree, first) is True
    assert controller.tree_installations == 1
    dispatch = controller.dispatch(first)
    assert dispatch.action == (0,)
    assert dispatch.shadow_recommended_action == (1,)
    assert dispatch.action_authority_enabled is False
    assert dispatch.mode == "phase1_direct_with_valid_shadow_tree"

    blocked = controller.dispatch(first)
    assert blocked.action is None
    assert blocked.mode == "blocked_real_observation_required"

    observed = controller.observe(second)
    assert observed.state is ControllerState.READY
    assert observed.reused_subtree is True
    assert observed.reason == "deterministic_subtree_reused"
    assert controller.tree_installations == 1
    assert controller.subtree_reuses == 1
    assert observed.tree_sha256 == tree.tree_sha256

    second_dispatch = controller.dispatch(second)
    assert second_dispatch.action == ()
    assert second_dispatch.shadow_recommended_action == (2,)


def test_observation_legal_or_option_fingerprint_mismatch_forces_rebuild() -> None:
    config, first, second, tree = _two_step_cross_turn_tree()
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(tree, first)
    controller.dispatch(first)

    stale_key = PublicStateKey(
        **{
            **second.key.as_payload(),
            "turn_key": second.key.turn_key,
            "observation_sha256": _digest("unexpected-observation"),
        }
    )
    stale = DecisionSnapshot(
        key=stale_key,
        legal_actions=second.legal_actions,
        direct_action=second.direct_action,
    )
    result = controller.observe(stale)
    assert result.reused_subtree is False
    assert result.rebuild_required is True
    assert result.reason == "cached_child_fingerprint_mismatch"
    assert result.tree_sha256 is None


@pytest.mark.parametrize("label", ["true", "false"])
def test_public_branch_selects_only_an_explicit_real_outcome(label: str) -> None:
    config = ChanceAwareSearchConfig()
    root_snapshot = _snapshot(
        "branch-root",
        turn_key=(1, 4),
        serial=20,
        legal_actions=((4,),),
        direct_action=(4,),
    )
    true_snapshot = _snapshot(
        "branch-true",
        turn_key=(1, 4),
        serial=21,
        legal_actions=((5,),),
        direct_action=(5,),
    )
    false_snapshot = _snapshot(
        "branch-false",
        turn_key=(1, 4),
        serial=21,
        legal_actions=((6,),),
        direct_action=(6,),
    )
    terminal = TerminalNode(Fraction(0))
    true_child = _decision(
        true_snapshot,
        recommended=(5,),
        transitions={(5,): TerminalTransition(terminal)},
    )
    false_child = _decision(
        false_snapshot,
        recommended=(6,),
        transitions={(6,): TerminalTransition(terminal)},
    )
    branch = ObservedPublicBranchTransition(
        predicate_id="target_found",
        certificate_sha256=_digest("branch-certificate"),
        outcomes=(
            BranchOutcome("true", true_child),
            BranchOutcome("false", false_child),
        ),
    )
    root = _decision(
        root_snapshot,
        recommended=(4,),
        transitions={(4,): branch},
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), root_snapshot)
    controller.dispatch(root_snapshot)

    expected = true_snapshot if label == "true" else false_snapshot
    result = controller.observe(
        expected,
        branch_evidence=ObservedPublicBranchEvidence(
            predicate_id=branch.predicate_id,
            outcome_label=label,
            certificate_sha256=branch.certificate_sha256,
            observation_sha256=expected.key.observation_sha256,
            public_state_sha256=expected.key.identity_sha256,
        ),
    )
    assert result.reused_subtree is True
    assert result.reason == "attested_public_branch_subtree_reused"


def test_missing_public_branch_never_defaults_then_or_else() -> None:
    config = ChanceAwareSearchConfig()
    root_snapshot = _snapshot(
        "missing-branch-root",
        turn_key=(0, 2),
        serial=30,
        legal_actions=((7,),),
        direct_action=(7,),
    )
    child_snapshot = _snapshot(
        "missing-branch-child",
        turn_key=(0, 2),
        serial=31,
        legal_actions=((8,),),
        direct_action=(8,),
    )
    terminal = TerminalNode(Fraction(0))
    child = _decision(
        child_snapshot,
        recommended=(8,),
        transitions={(8,): TerminalTransition(terminal)},
    )
    branch = ObservedPublicBranchTransition(
        predicate_id="target_found",
        certificate_sha256=_digest("branch-certificate"),
        outcomes=(BranchOutcome("true", child), BranchOutcome("false", child)),
    )
    root = _decision(
        root_snapshot,
        recommended=(7,),
        transitions={(7,): branch},
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), root_snapshot)
    controller.dispatch(root_snapshot)

    result = controller.observe(child_snapshot, branch_evidence=None)
    assert result.reused_subtree is False
    assert result.rebuild_required is True
    assert result.reason == "missing_or_unknown_public_branch_outcome"


def test_exact_fair_coin_uses_fractional_expectation_but_rebuilds_when_realized() -> None:
    chance = FiniteChanceNode(
        event_id="fair_coin",
        distribution_receipt_sha256=_digest("fair-coin-distribution"),
        outcomes=(
            ChanceOutcome("heads", Fraction(1, 2), TerminalNode(Fraction(1))),
            ChanceOutcome("tails", Fraction(1, 2), TerminalNode(Fraction(-1))),
        ),
    )
    assert chance.expected_value == Fraction(0)

    config = ChanceAwareSearchConfig()
    first = _snapshot(
        "coin-root",
        turn_key=(0, 9),
        serial=40,
        legal_actions=((9,),),
        direct_action=(9,),
    )
    after = _snapshot(
        "after-coin",
        turn_key=(0, 9),
        serial=41,
        legal_actions=((10,),),
        direct_action=(10,),
    )
    root = _decision(
        first,
        recommended=(9,),
        transitions={(9,): FiniteChanceTransition(chance)},
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), first)
    controller.dispatch(first)

    result = controller.observe(after, chance_outcome_label="heads")
    assert result.chance_expected_value == Fraction(0)
    assert result.reused_subtree is False
    assert result.rebuild_required is True
    assert result.reason == "realized_finite_chance_starts_fresh_public_root"


@pytest.mark.parametrize(
    "outcomes, message",
    [
        (
            (
                ChanceOutcome("a", Fraction(1, 3), TerminalNode(Fraction(0))),
                ChanceOutcome("b", Fraction(1, 3), TerminalNode(Fraction(0))),
            ),
            "sum exactly",
        ),
        (
            (
                ChanceOutcome("same", Fraction(1, 2), TerminalNode(Fraction(0))),
                ChanceOutcome("same", Fraction(1, 2), TerminalNode(Fraction(0))),
            ),
            "unique",
        ),
    ],
)
def test_invalid_finite_chance_distribution_is_rejected(
    outcomes: tuple[ChanceOutcome, ...],
    message: str,
) -> None:
    with pytest.raises(ChanceAwareTreeError, match=message):
        FiniteChanceNode(
            event_id="invalid",
            distribution_receipt_sha256=_digest("invalid"),
            outcomes=outcomes,
        )


def test_unknown_or_opponent_information_boundary_requires_rebuild() -> None:
    config = ChanceAwareSearchConfig()
    first = _snapshot(
        "boundary-root",
        turn_key=(0, 12),
        serial=50,
        legal_actions=((11,),),
        direct_action=(11,),
    )
    after = _snapshot(
        "boundary-after",
        turn_key=(1, 12),
        serial=51,
        legal_actions=((12,),),
        direct_action=(12,),
    )
    root = _decision(
        first,
        recommended=(11,),
        transitions={
            (11,): RebuildBoundaryTransition(
                "opponent_or_private_information_without_receipted_distribution"
            )
        },
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), first)
    controller.dispatch(first)

    result = controller.observe(after)
    assert result.rebuild_required is True
    assert result.reason.startswith("rebuild_boundary:opponent_or_private")


def test_phase1_always_uses_exact_direct_action_and_preserves_empty_action() -> None:
    config = ChanceAwareSearchConfig()
    snapshot = _snapshot(
        "empty-action",
        turn_key=(0, 3),
        serial=60,
        legal_actions=((), (13,)),
        direct_action=(),
    )
    terminal = TerminalNode(Fraction(0))
    root = _decision(
        snapshot,
        recommended=(13,),
        transitions={
            (): TerminalTransition(terminal),
            (13,): TerminalTransition(terminal),
        },
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), snapshot)

    result = controller.dispatch(snapshot)
    assert result.action == ()
    assert result.direct_action == ()
    assert result.shadow_recommended_action == (13,)
    assert result.action_authority_enabled is False


def test_direct_action_edge_is_mandatory_and_over_cap_never_truncates() -> None:
    config = ChanceAwareSearchConfig(max_complete_actions=2)
    snapshot = _snapshot(
        "mandatory-direct",
        turn_key=(0, 4),
        serial=70,
        legal_actions=((0,), (1,), (2,)),
        direct_action=(0,),
    )
    terminal = TerminalNode(Fraction(0))
    with pytest.raises(ChanceAwareTreeError, match="direct action requires"):
        _decision(
            snapshot,
            recommended=(1,),
            transitions={(1,): TerminalTransition(terminal)},
        )

    root = _decision(
        snapshot,
        recommended=(1,),
        transitions={
            (0,): TerminalTransition(terminal),
            (1,): TerminalTransition(terminal),
        },
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), snapshot) is False
    dispatch = controller.dispatch(snapshot)
    assert dispatch.action == (0,)
    assert "complete_action_cap_exceeded" in dispatch.reasons


def test_step_and_turn_timeouts_discard_tree_and_return_direct() -> None:
    step_config, first, _second, tree = _two_step_cross_turn_tree()
    step_controller = ChanceAwareTreeController(
        step_config,
        clock=_ScriptedClock(0.0, 5.01, 5.01, 5.01),
    )
    assert step_controller.install_tree(tree, first) is False
    step_dispatch = step_controller.dispatch(first)
    assert step_dispatch.action == first.direct_action
    assert "planner_budget_exhausted" in step_dispatch.reasons
    assert step_dispatch.tree_sha256 is None

    turn_config = ChanceAwareSearchConfig(
        max_turn_seconds=6.0,
        max_action_seconds=5.0,
    )
    _, same_first, cross_second, _ = _two_step_cross_turn_tree(turn_config)
    same_second = _snapshot(
        "same-turn-second",
        turn_key=same_first.key.turn_key,
        serial=cross_second.key.decision_serial,
        legal_actions=cross_second.legal_actions,
        direct_action=cross_second.direct_action,
    )
    terminal = TerminalNode(Fraction(0))
    child = _decision(
        same_second,
        recommended=(2,),
        transitions={
            (): TerminalTransition(terminal),
            (2,): TerminalTransition(terminal),
        },
    )
    root = _decision(
        same_first,
        recommended=(1,),
        transitions={
            (0,): DeterministicTransition(_digest("same-turn"), child),
            (1,): RebuildBoundaryTransition("unexpanded"),
        },
    )
    controller = ChanceAwareTreeController(
        turn_config,
        clock=_ScriptedClock(0.0, 4.0, 4.0, 5.0, 5.0, 7.0),
    )
    assert controller.install_tree(_tree(root, turn_config), same_first)
    controller.dispatch(same_first)
    result = controller.observe(same_second)
    assert result.rebuild_required is True
    assert result.reason == "planner_budget_exhausted_during_observation_validation"
    assert result.budget.turn_exhausted is True


def test_protocol_rejects_observation_without_dispatch_and_tree_is_immutable() -> None:
    config, first, second, tree = _two_step_cross_turn_tree()
    controller = ChanceAwareTreeController(config)
    original_digest = tree.tree_sha256
    with pytest.raises(ControllerProtocolError, match="only after"):
        controller.observe(second)
    assert controller.install_tree(tree, first)
    assert tree.tree_sha256 == original_digest

    with pytest.raises(FrozenInstanceError):
        tree.root.edges += ()  # type: ignore[misc]


def test_predicted_terminal_never_overrides_a_real_nonterminal_snapshot() -> None:
    config = ChanceAwareSearchConfig()
    first = _snapshot(
        "predicted-terminal-root",
        turn_key=(0, 15),
        serial=80,
        legal_actions=((1,),),
        direct_action=(1,),
    )
    next_snapshot = _snapshot(
        "reality-continued",
        turn_key=(0, 15),
        serial=81,
        legal_actions=((2,),),
        direct_action=(2,),
    )
    root = _decision(
        first,
        recommended=(1,),
        transitions={
            (1,): TerminalTransition(TerminalNode(Fraction(1), "predicted"))
        },
    )
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), first)
    controller.dispatch(first)

    observed = controller.observe(next_snapshot)
    assert observed.state is ControllerState.REBUILD_REQUIRED
    assert observed.terminated is False
    assert observed.reason == "predicted_terminal_not_confirmed"
    fallback = controller.dispatch(next_snapshot)
    assert fallback.action == next_snapshot.direct_action
    assert fallback.mode == "exact_direct_fallback"


def test_rebuild_install_rejects_replayed_or_different_snapshot_identity() -> None:
    config, first, second, tree = _two_step_cross_turn_tree()
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(tree, first)
    controller.dispatch(first)

    changed_key = PublicStateKey(
        **{
            **second.key.as_payload(),
            "turn_key": second.key.turn_key,
            "observation_sha256": _digest("changed-reality"),
        }
    )
    changed = DecisionSnapshot(
        key=changed_key,
        legal_actions=second.legal_actions,
        direct_action=second.direct_action,
    )
    assert controller.observe(changed).rebuild_required is True

    old_root = _decision(
        first,
        recommended=(1,),
        transitions={
            (0,): RebuildBoundaryTransition("old"),
            (1,): RebuildBoundaryTransition("old"),
        },
    )
    assert controller.install_tree(_tree(old_root, config), first) is False

    wrong_same_serial = _decision(
        second,
        recommended=(2,),
        transitions={
            (): RebuildBoundaryTransition("wrong"),
            (2,): RebuildBoundaryTransition("wrong"),
        },
    )
    assert controller.install_tree(_tree(wrong_same_serial, config), second) is False

    current_root = _decision(
        changed,
        recommended=(2,),
        transitions={
            (): RebuildBoundaryTransition("current"),
            (2,): RebuildBoundaryTransition("current"),
        },
    )
    assert controller.install_tree(_tree(current_root, config), changed) is True


def test_tree_and_config_digest_drift_fail_closed_to_direct_action() -> None:
    config, first, _second, tree = _two_step_cross_turn_tree()
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(tree, first)
    object.__setattr__(tree.root, "search_value", Fraction(99))

    result = controller.dispatch(first)
    assert result.action == first.direct_action
    assert result.mode == "exact_direct_fallback"
    assert "cached_tree_content_digest_changed" in result.reasons

    config2, first2, _second2, tree2 = _two_step_cross_turn_tree()
    controller2 = ChanceAwareTreeController(config2)
    assert controller2.install_tree(tree2, first2)
    with pytest.raises(AttributeError):
        controller2.config = ChanceAwareSearchConfig()  # type: ignore[misc]
    object.__setattr__(config2, "max_action_seconds", 4.0)
    result2 = controller2.dispatch(first2)
    assert result2.action == first2.direct_action
    assert "planner_config_identity_changed" in result2.reasons


def test_public_branch_evidence_is_bound_to_predicate_certificate_and_snapshot() -> None:
    config = ChanceAwareSearchConfig()
    first = _snapshot(
        "bound-branch-root",
        turn_key=(0, 16),
        serial=90,
        legal_actions=((3,),),
        direct_action=(3,),
    )
    child_snapshot = _snapshot(
        "bound-branch-child",
        turn_key=(0, 16),
        serial=91,
        legal_actions=((4,),),
        direct_action=(4,),
    )
    child = _decision(
        child_snapshot,
        recommended=(4,),
        transitions={(4,): RebuildBoundaryTransition("done")},
    )
    branch = ObservedPublicBranchTransition(
        predicate_id="public_predicate",
        certificate_sha256=_digest("public-certificate"),
        outcomes=(BranchOutcome("true", child), BranchOutcome("false", child)),
    )
    root = _decision(first, recommended=(3,), transitions={(3,): branch})
    controller = ChanceAwareTreeController(config)
    assert controller.install_tree(_tree(root, config), first)
    controller.dispatch(first)

    result = controller.observe(
        child_snapshot,
        branch_evidence=ObservedPublicBranchEvidence(
            predicate_id="wrong_predicate",
            outcome_label="true",
            certificate_sha256=branch.certificate_sha256,
            observation_sha256=child_snapshot.key.observation_sha256,
            public_state_sha256=child_snapshot.key.identity_sha256,
        ),
    )
    assert result.rebuild_required is True
    assert result.reason == "public_branch_evidence_mismatch"


def test_nested_exact_chance_values_are_supported_without_sampling() -> None:
    inner = FiniteChanceNode(
        event_id="inner",
        distribution_receipt_sha256=_digest("inner"),
        outcomes=(
            ChanceOutcome("a", Fraction(1, 4), TerminalNode(Fraction(4))),
            ChanceOutcome("b", Fraction(3, 4), TerminalNode(Fraction(0))),
        ),
    )
    outer = FiniteChanceNode(
        event_id="outer",
        distribution_receipt_sha256=_digest("outer"),
        outcomes=(
            ChanceOutcome("heads", Fraction(1, 2), inner),
            ChanceOutcome("tails", Fraction(1, 2), TerminalNode(Fraction(-1))),
        ),
    )
    assert inner.expected_value == Fraction(1)
    assert outer.expected_value == Fraction(0)


def test_terminal_root_and_caller_declared_elapsed_time_are_rejected() -> None:
    config, first, _second, tree = _two_step_cross_turn_tree()
    with pytest.raises(ChanceAwareTreeError, match="root must be a decision"):
        CachedActionTree(  # type: ignore[arg-type]
            root=TerminalNode(Fraction(0)),
            config_sha256=config.identity_sha256,
            planner_artifact_sha256=_digest("bad-root"),
        )
    controller = ChanceAwareTreeController(config)
    with pytest.raises(TypeError):
        controller.install_tree(tree, first, planner_seconds=0.0)  # type: ignore[call-arg]


def test_normal_submodule_import_does_not_load_torch_bridge_or_executor() -> None:
    code = """
import sys
import poke_bot.recursive_turn_planner.chance_aware_tree
for name in (
    'torch',
    'poke_bot.recursive_turn_planner.agent_bridge',
    'poke_bot.recursive_turn_planner.executor',
    'poke_bot.recursive_turn_planner.planner',
):
    assert name not in sys.modules, name
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_module_has_no_engine_model_rng_service_or_legacy_executor_import() -> None:
    parsed = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "torch",
        "random",
        "subprocess",
        "poke_bot.cg_env",
        "poke_bot.agent",
        "poke_bot.model",
        "poke_bot.recursive_turn_planner.executor",
        "poke_bot.recursive_turn_planner.memory",
    }
    assert imported.isdisjoint(forbidden)
