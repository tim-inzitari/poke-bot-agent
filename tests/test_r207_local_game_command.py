"""Hermetic integration coverage for the r207 local MCTS game command."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from poke_bot.recursive_turn_planner.bo1000_evaluation import (
    R207_FROZEN_BUNDLE_SHA256,
    R207_FROZEN_CHECKPOINT_SHA256,
    BO1000GameReceipt,
    build_bo1000_schedule,
    compile_bo1000_report,
)
from poke_bot.recursive_turn_planner.bo1000_pair_runner import (
    HostLocalBO1000PairController,
    build_bo1000_pair_envelope,
    build_r207_local_game_command,
)
from poke_bot.recursive_turn_planner.chance_aware_tree import ChanceAwareSearchConfig
from poke_bot.recursive_turn_planner.neural_leaf_reranker import (
    BatchedNeuralLeafReranker,
    FrozenPolicyIdentity,
    NeuralForwardRow,
    NeuralLeafRerankerConfig,
)
from poke_bot.recursive_turn_planner.r207_frozen_leaf_calibration import (
    R195_CHECKPOINT_BYTES,
    R195_CHECKPOINT_SHA256,
    R207_FROZEN_PREDICTIONS_SCHEMA,
    R207_HELDOUT_EVIDENCE_SCHEMA,
    R207_TRAINING_SOURCE_MANIFEST_SCHEMA,
    FrozenLeafCalibrationPolicy,
    canonical_sha256,
    compile_r207_frozen_leaf_calibration_preflight,
)
from poke_bot.recursive_turn_planner.r207_local_game_command import (
    R207_REAL_SUCCESSOR_INTERFACE_KIND,
    R207_R195_MATCHUP_TREE_SHA256,
    R207AppliedAction,
    R207ArmRuntimeTelemetry,
    R207GameTerminal,
    R207FrozenLeafCalibrationBinding,
    R207LocalGameCommandAdapter,
    R207LocalGameRequest,
    R207MCTSRuntimeBinding,
    R207RealSuccessorBinding,
    R207RuntimeParityTelemetry,
    parse_r207_local_game_request,
)
from poke_bot.recursive_turn_planner.r207_simulator_arena import (
    OpaqueMidgameHandle,
    SuccessorTransition,
    TransitionKind,
    public_observation_sha256,
)
from poke_bot.recursive_turn_planner.simulator_one_turn_expectimax import (
    MCTSExpansionProfile,
    PolicyDecisionView,
    R207ArenaAdapter,
    SimulatorInterTurnMCTSSession,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


class _NativeArena:
    """Small real-interface-shaped arena, not the v2 replay arena."""

    r207_successor_interface_kind = R207_REAL_SUCCESSOR_INTERFACE_KIND
    r207_successor_abi_receipt_sha256 = _digest("native-successor-abi")
    r207_future_legality_receipt_sha256 = _digest("native-future-legality")
    r207_exact_chance_receipt_sha256 = _digest("native-exact-chance")
    r207_terminal_result_parity_receipt_sha256 = _digest("native-terminal-parity")
    r207_deterministic_receipt_sha256 = _digest("native-determinism")

    def __init__(self) -> None:
        self._root_observation = {"node": "root", "current": {"result": -1}}
        self.handle = OpaqueMidgameHandle(
            _digest("native-arena"),
            _digest("native-root"),
            public_observation_sha256(self._root_observation),
        )
        self.expand_calls: list[tuple[int, ...]] = []

    def capture_root(self, deadline):
        deadline.check("native_capture_root")
        return self.handle, dict(self._root_observation)

    def observe(self, handle, deadline):
        deadline.check("native_observe")
        assert handle == self.handle
        return dict(self._root_observation)

    def expand_action(self, handle, action, deadline):
        deadline.check("native_expand")
        assert handle == self.handle
        selected = tuple(action)
        self.expand_calls.append(selected)
        return SuccessorTransition(
            kind=TransitionKind.TERMINAL,
            transition_certificate_sha256=_digest(f"native-terminal:{selected}"),
            public_observation_sha256=_digest(f"native-terminal-public:{selected}"),
            terminal_result=0 if selected == (0,) else 1,
        )


class _PolicyFactory:
    def __init__(self, arena: _NativeArena) -> None:
        self._arena = arena

    def root_view(self) -> PolicyDecisionView:
        return PolicyDecisionView(
            handle=self._arena.handle,
            public_observation_sha256=self._arena.handle.public_observation_sha256,
            turn_key=(0, 0),
            decision_serial=0,
            acting_seat=0,
            legal_actions=((0,), (1,)),
            option_encoding_sha256=_digest("root-options"),
            direct_action=(1,),
            packet={"fake": "policy-visible-root"},
            future_legality_receipt_sha256=_digest("future-legality"),
            simulator_result_sha256=_digest("simulator-root"),
        )

    def direct_view(self) -> PolicyDecisionView:
        return PolicyDecisionView(
            handle=self._arena.handle,
            public_observation_sha256=self._arena.handle.public_observation_sha256,
            turn_key=(1, 0),
            decision_serial=1,
            acting_seat=1,
            legal_actions=((9,),),
            option_encoding_sha256=_digest("direct-options"),
            direct_action=(9,),
            packet={"fake": "policy-visible-direct"},
            future_legality_receipt_sha256=_digest("direct-future-legality"),
            simulator_result_sha256=_digest("direct-simulator"),
        )

    def build_decision(self, **_kwargs) -> PolicyDecisionView:
        return self.root_view()

    def build_boundary_leaf(self, **_kwargs) -> None:
        return None


class _FrozenBackend:
    def __init__(self, binding: R207MCTSRuntimeBinding) -> None:
        self.frozen_policy_identity_sha256 = binding.frozen_policy_identity.identity_sha256
        self.r207_no_rtp_runtime_receipt_sha256 = binding.no_rtp_runtime_receipt_sha256
        self.r207_direct_runtime_graph_sha256 = binding.direct_runtime_graph_sha256
        self.r207_matchup_tree_sha256 = binding.matchup_tree_sha256
        self.r207_matchup_adapter_bank_sha256 = binding.matchup_adapter_bank_sha256
        self.r207_matchup_adapter_training_receipt_sha256 = (
            binding.matchup_adapter_training_receipt_sha256
        )
        self.r207_matchup_adapter_runtime_graph_sha256 = (
            binding.matchup_adapter_runtime_graph_sha256
        )
        self.r207_matchup_adapter_enabled = True
        self.r207_matchup_adapter_trained = True
        self.r207_matchup_adapter_frozen = True
        self.r207_guide2vec_enabled = False
        self.r207_guide_logit_transform_enabled = False
        self.r207_guide_linear_transform_enabled = False
        self.r207_legacy_rtp_sidecar_used = False
        self.r207_legacy_rtp_executor_used = False
        self.r207_leaf_calibration_receipt_sha256 = (
            binding.calibration.leaf_calibration.receipt_sha256
        )
        self.r207_leaf_calibration_receipt_file_sha256 = (
            binding.calibration.receipt_file_sha256
        )

    def __call__(self, requests) -> Sequence[NeuralForwardRow]:
        return tuple(
            NeuralForwardRow(
                policy_logits=(0.0, 0.0),
                value_from_actor=0.0,
                acting_seat=0,
                outcome_distribution_logits_from_actor=(0.0, 0.0, 0.0),
            )
            for _request in requests
        )


def _write_sealed_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    os.chmod(path, 0o444)


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_calibration_binding() -> R207FrozenLeafCalibrationBinding:
    """Build a tiny, genuine sealed r207 calibration receipt for this test."""

    # The immutable reader rejects the macOS /var -> /private/var symlink, so
    # keep these hermetic sealed inputs directly below the real workspace root.
    with tempfile.TemporaryDirectory(
        prefix="r207-local-calibration-", dir=Path.cwd()
    ) as directory:
        root = Path(directory)
        training_path = root / "r195-training-source-manifest.json"
        training_source_ids = ["training-a"]
        training_game_ids = ["training-game-a"]
        _write_sealed_json(
            training_path,
            {
                "schema": R207_TRAINING_SOURCE_MANIFEST_SCHEMA,
                "status": "sealed",
                "checkpoint_sha256": R195_CHECKPOINT_SHA256,
                "checkpoint_bytes": R195_CHECKPOINT_BYTES,
                "source_role": "r195_training",
                "training_eligible": True,
                "attempt10_rtp_partial_rows_used": False,
                "r197_or_r198_rtp_partial_rows_used": False,
                "r195_rtp_sidecar_used": False,
                "r197_rtp_sidecar_used": False,
                "training_source_ids": training_source_ids,
                "training_source_ids_sha256": canonical_sha256(training_source_ids),
                "training_game_ids": training_game_ids,
                "training_game_ids_sha256": canonical_sha256(training_game_ids),
            },
        )
        heldout_source_ids = ["heldout-a"]
        heldout_game_ids = ["heldout-game-01", "heldout-game-02", "heldout-game-03"]
        leaves = (
            ("leaf-01", "heldout-game-01", "loss", (0.98, 0.01, 0.01), -0.95),
            ("leaf-02", "heldout-game-02", "draw", (0.01, 0.98, 0.01), 0.0),
            ("leaf-03", "heldout-game-03", "win", (0.01, 0.01, 0.98), 0.95),
        )
        evidence_path = root / "r207-heldout-evidence.json"
        evidence = {
            "schema": R207_HELDOUT_EVIDENCE_SCHEMA,
            "status": "sealed",
            "evaluation_only": True,
            "training_eligible": False,
            "replay_eligible": False,
            "evidence_role": "r207_source_excluded_terminal_leaf_calibration",
            "attempt10_rtp_partial_rows_used": False,
            "r197_or_r198_rtp_partial_rows_used": False,
            "r195_rtp_sidecar_used": False,
            "r197_rtp_sidecar_used": False,
            "outcome_perspective": "candidate_root",
            "terminal_outcomes_exact": True,
            "nonterminal_policy_visible_leaf_predictions_only": True,
            "source_exclusion": {
                "source_disjoint": True,
                "game_disjoint": True,
                "r195_training_source_manifest_sha256": _file_sha256(training_path),
                "training_source_ids_sha256": canonical_sha256(training_source_ids),
                "training_game_ids_sha256": canonical_sha256(training_game_ids),
                "heldout_source_ids": heldout_source_ids,
                "heldout_game_ids": heldout_game_ids,
                "heldout_source_ids_sha256": canonical_sha256(heldout_source_ids),
                "heldout_game_ids_sha256": canonical_sha256(heldout_game_ids),
                "intersection_source_ids_sha256": canonical_sha256([]),
                "intersection_source_id_count": 0,
                "intersection_game_ids_sha256": canonical_sha256([]),
                "intersection_game_id_count": 0,
            },
            "terminal_leaf_count": len(leaves),
            "terminal_leaves": [
                {
                    "leaf_id": leaf_id,
                    "source_id": "heldout-a",
                    "game_id": game_id,
                    "decision_ordinal": 1,
                    "root_seat": 0,
                    "horizon_atomic_actions": 1,
                    "leaf_kind": "successor",
                    "policy_visible": True,
                    "is_terminal": False,
                    "public_observation_sha256": _digest(f"observation:{leaf_id}"),
                    "legal_actions_sha256": _digest(f"legal-actions:{leaf_id}"),
                    "nonterminal_leaf_state_sha256": _digest(f"leaf-state:{leaf_id}"),
                    "terminal_outcome": outcome,
                    "terminal_result_kind": "exact_simulator_terminal",
                    "terminal_result_sha256": _digest(f"terminal:{leaf_id}"),
                }
                for leaf_id, game_id, outcome, _probabilities, _value in leaves
            ],
        }
        _write_sealed_json(evidence_path, evidence)
        identity = FrozenPolicyIdentity.r205_no_rtp()
        predictions_path = root / "r207-frozen-predictions.json"
        _write_sealed_json(
            predictions_path,
            {
                "schema": R207_FROZEN_PREDICTIONS_SCHEMA,
                "status": "frozen_inference_complete",
                "outcome_perspective": "candidate_root",
                "outcome_class_order": ["loss", "draw", "win"],
                "model_role": "r195_no_rtp_frozen_policy_and_leaf_model",
                "frozen_policy_identity_sha256": identity.identity_sha256,
                "r195_no_rtp_bundle_sha256": identity.bundle_sha256,
                "r195_no_rtp_deck_cards_sha256": identity.deck_cards_sha256,
                "r195_no_rtp_runtime_sha256": identity.nonplanner_runtime_sha256,
                "attempt10_rtp_partial_rows_used": False,
                "r197_or_r198_rtp_partial_rows_used": False,
                "r195_rtp_sidecar_used": False,
                "r197_rtp_sidecar_used": False,
                "checkpoint_sha256": R195_CHECKPOINT_SHA256,
                "checkpoint_bytes": R195_CHECKPOINT_BYTES,
                "heldout_evidence_sha256": _file_sha256(evidence_path),
                "model_frozen": True,
                "training_performed": False,
                "optimizer_steps": 0,
                "gradient_updates": 0,
                "training_eligible": False,
                "predictions": [
                    {
                        "leaf_id": leaf_id,
                        "source_id": "heldout-a",
                        "game_id": game_id,
                        "decision_ordinal": 1,
                        "root_seat": 0,
                        "horizon_atomic_actions": 1,
                        "leaf_kind": "successor",
                        "public_observation_sha256": _digest(f"observation:{leaf_id}"),
                        "legal_actions_sha256": _digest(f"legal-actions:{leaf_id}"),
                        "nonterminal_leaf_state_sha256": _digest(f"leaf-state:{leaf_id}"),
                        "terminal_result_sha256": _digest(f"terminal:{leaf_id}"),
                        "outcome_probabilities": {
                            "loss": probabilities[0],
                            "draw": probabilities[1],
                            "win": probabilities[2],
                        },
                        "value": value,
                    }
                    for leaf_id, game_id, _outcome, probabilities, value in leaves
                ],
            },
        )
        policy = FrozenLeafCalibrationPolicy(
            min_terminal_leaves=3,
            min_distinct_source_ids=1,
            min_distinct_game_ids=3,
            min_per_outcome=1,
            min_per_source_id=1,
            max_outcome_brier=1.0,
            max_outcome_ece=1.0,
            max_value_rmse=2.0,
            max_value_mae=2.0,
            value_weight=0.4,
            outcome_weight=0.6,
            uncertainty_quantile=0.95,
            ece_bins=3,
        )
        calibration_receipt = compile_r207_frozen_leaf_calibration_preflight(
            evidence_path,
            predictions_path,
            training_path,
            policy=policy,
        )
        receipt_path = root / "r207-frozen-leaf-calibration-receipt.json"
        _write_sealed_json(receipt_path, calibration_receipt)
        return R207FrozenLeafCalibrationBinding(
            calibration_receipt_path=receipt_path,
            heldout_evidence_path=evidence_path,
            frozen_predictions_path=predictions_path,
            r195_training_source_manifest_path=training_path,
        )


def _binding(arena: _NativeArena) -> R207MCTSRuntimeBinding:
    return R207MCTSRuntimeBinding(
        successor=R207RealSuccessorBinding(
            arena=arena,
            interface_kind=R207_REAL_SUCCESSOR_INTERFACE_KIND,
            successor_abi_receipt_sha256=arena.r207_successor_abi_receipt_sha256,
            future_legality_receipt_sha256=_digest("native-future-legality"),
            exact_chance_receipt_sha256=_digest("native-exact-chance"),
            terminal_result_parity_receipt_sha256=_digest("native-terminal-parity"),
            deterministic_receipt_sha256=_digest("native-determinism"),
        ),
        calibration=_verified_calibration_binding(),
        frozen_policy_identity=FrozenPolicyIdentity.r205_no_rtp(),
        no_rtp_runtime_receipt_sha256=_digest("r195-no-rtp-runtime"),
        matchup_tree_sha256=R207_R195_MATCHUP_TREE_SHA256,
        matchup_adapter_bank_sha256=_digest("r195-adapter-bank"),
        matchup_adapter_training_receipt_sha256=_digest("r195-adapter-training"),
        matchup_adapter_runtime_graph_sha256=_digest("r195-adapter-runtime"),
        direct_runtime_graph_sha256=_digest("r195-direct-runtime"),
    )


def _session(
    arena: _NativeArena,
    factory: _PolicyFactory,
    binding: R207MCTSRuntimeBinding,
    *,
    config: ChanceAwareSearchConfig | None = None,
    calibrated: bool = True,
) -> SimulatorInterTurnMCTSSession:
    reranker = BatchedNeuralLeafReranker(
        _FrozenBackend(binding),
        NeuralLeafRerankerConfig(
            frozen_policy_identity=binding.frozen_policy_identity,
            calibration=binding.calibration.leaf_calibration if calibrated else None,
        ),
    )
    return SimulatorInterTurnMCTSSession(
        adapter=R207ArenaAdapter(arena, factory),
        reranker=reranker,
        config=config or ChanceAwareSearchConfig(),
        profile=MCTSExpansionProfile(
            requested_simulations=2,
            max_decision_depth=1,
            max_tree_nodes=8,
        ),
    )


class _LiveGame:
    def __init__(
        self,
        *,
        arena: _NativeArena,
        factory: _PolicyFactory,
        binding: R207MCTSRuntimeBinding,
        session: SimulatorInterTurnMCTSSession,
        parity: R207RuntimeParityTelemetry | None = None,
    ) -> None:
        self.prerequisites = binding
        self.mcts_session = session
        self._factory = factory
        self._phase = 0
        self.actions: list[tuple[int, tuple[int, ...]]] = []
        self._parity = parity or _parity(binding)

    def next_decision(self) -> PolicyDecisionView | None:
        if self._phase == 0:
            return self._factory.root_view()
        if self._phase == 1:
            return self._factory.direct_view()
        return None

    def apply_action(self, action: Sequence[int]) -> R207AppliedAction:
        view = self.next_decision()
        assert view is not None
        selected = tuple(action)
        assert selected in view.legal_actions
        self.actions.append((view.acting_seat, selected))
        self._phase += 1
        return R207AppliedAction(
            realized_transition_kind=(
                TransitionKind.DETERMINISTIC_PUBLIC if view.acting_seat == 0 else None
            ),
            next_mcts_view=None,
        )

    def terminal(self) -> R207GameTerminal:
        assert self._phase == 2
        return R207GameTerminal(winner_seat=0)

    def runtime_parity_telemetry(self) -> R207RuntimeParityTelemetry:
        return self._parity


def _parity(binding: R207MCTSRuntimeBinding) -> R207RuntimeParityTelemetry:
    arm = R207ArmRuntimeTelemetry(
        frozen_policy_identity_sha256=binding.frozen_policy_identity.identity_sha256,
        no_rtp_runtime_receipt_sha256=binding.no_rtp_runtime_receipt_sha256,
        direct_runtime_graph_sha256=binding.direct_runtime_graph_sha256,
        matchup_tree_sha256=binding.matchup_tree_sha256,
        matchup_adapter_bank_sha256=binding.matchup_adapter_bank_sha256,
        matchup_adapter_training_receipt_sha256=(
            binding.matchup_adapter_training_receipt_sha256
        ),
        matchup_adapter_runtime_graph_sha256=binding.matchup_adapter_runtime_graph_sha256,
        matchup_adapter_enabled=True,
        matchup_adapter_trained=True,
        matchup_adapter_frozen=True,
        guide2vec_enabled=False,
        guide_logit_transform_enabled=False,
        guide_linear_transform_enabled=False,
        legacy_rtp_sidecar_used=False,
        legacy_rtp_executor_used=False,
    )
    return R207RuntimeParityTelemetry(mcts_arm=arm, direct_arm=arm)


def _request() -> R207LocalGameRequest:
    schedule = build_bo1000_schedule(_digest("local-command-schedule"))
    envelope = build_bo1000_pair_envelope(
        schedule,
        pair_index=0,
        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
        pair_rng_snapshot_sha256=_digest("pair-rng"),
        deck_order_rng_sha256=_digest("deck-rng"),
        execution_host="hermetic-local-host",
    )
    controller = HostLocalBO1000PairController(
        "/tmp/r207-local-game-command-unused", execution_host="hermetic-local-host"
    )
    return parse_r207_local_game_request(
        controller._game_request_payload(envelope, envelope.game_specs[0])
    )


def _live_factory(created: list[_LiveGame]):
    def factory(_request: R207LocalGameRequest) -> _LiveGame:
        arena = _NativeArena()
        policy_factory = _PolicyFactory(arena)
        binding = _binding(arena)
        game = _LiveGame(
            arena=arena,
            factory=policy_factory,
            binding=binding,
            session=_session(arena, policy_factory, binding),
        )
        created.append(game)
        return game

    return factory


def test_local_game_command_wraps_exact_r195_runtime_and_records_real_mcts_telemetry() -> None:
    request = _request()
    created: list[_LiveGame] = []
    receipt = R207LocalGameCommandAdapter(_live_factory(created)).execute(request)

    assert receipt.terminal_status == "completed"
    assert receipt.winner_seat == 0
    assert created[0].actions == [(0, (0,)), (1, (9,))]
    assert len(receipt.mcts_turns) == 1
    turn = receipt.mcts_turns[0]
    assert turn.mcts_seat == 0
    assert turn.simulator_transitions_seen >= 2
    assert turn.terminal_exact_results_seen >= 1
    assert turn.frozen_policy_prior_evaluations >= 1
    assert turn.frozen_outcome_leaf_evaluations >= 1
    assert turn.frozen_value_leaf_evaluations >= 1
    assert turn.turn_planner_wall_seconds <= 20.0
    assert turn.max_single_action_planner_wall_seconds <= 5.0


def test_local_game_command_fails_closed_without_calibration_or_exact_clock() -> None:
    request = _request()

    def missing_calibration(_request: R207LocalGameRequest) -> _LiveGame:
        arena = _NativeArena()
        policy_factory = _PolicyFactory(arena)
        binding = _binding(arena)
        return _LiveGame(
            arena=arena,
            factory=policy_factory,
            binding=binding,
            session=_session(arena, policy_factory, binding, calibrated=False),
        )

    def wrong_clock(_request: R207LocalGameRequest) -> _LiveGame:
        arena = _NativeArena()
        policy_factory = _PolicyFactory(arena)
        binding = _binding(arena)
        return _LiveGame(
            arena=arena,
            factory=policy_factory,
            binding=binding,
            session=_session(
                arena,
                policy_factory,
                binding,
                config=ChanceAwareSearchConfig(max_turn_seconds=19.0, max_action_seconds=5.0),
            ),
        )

    for factory in (missing_calibration, wrong_clock):
        receipt = R207LocalGameCommandAdapter(factory).execute(request)
        assert receipt.terminal_status == "failed_closed"
        assert receipt.crash_count == 1


def test_local_game_command_fails_closed_without_the_bound_real_successor() -> None:
    request = _request()

    def wrong_successor(_request: R207LocalGameRequest) -> _LiveGame:
        bound_arena = _NativeArena()
        live_arena = _NativeArena()
        policy_factory = _PolicyFactory(live_arena)
        binding = _binding(bound_arena)
        return _LiveGame(
            arena=live_arena,
            factory=policy_factory,
            binding=binding,
            session=_session(live_arena, policy_factory, binding),
        )

    receipt = R207LocalGameCommandAdapter(wrong_successor).execute(request)
    assert receipt.terminal_status == "failed_closed"
    assert receipt.crash_count == 1


def test_local_game_command_fails_closed_when_arm_runtime_or_adapter_identity_differs() -> None:
    request = _request()

    def mismatched_runtime(_request: R207LocalGameRequest) -> _LiveGame:
        arena = _NativeArena()
        policy_factory = _PolicyFactory(arena)
        binding = _binding(arena)
        arm = _parity(binding).mcts_arm
        parity = R207RuntimeParityTelemetry(
            mcts_arm=arm,
            direct_arm=replace(arm, direct_runtime_graph_sha256=_digest("wrong-direct-runtime")),
        )
        return _LiveGame(
            arena=arena,
            factory=policy_factory,
            binding=binding,
            session=_session(arena, policy_factory, binding),
            parity=parity,
        )

    receipt = R207LocalGameCommandAdapter(mismatched_runtime).execute(request)
    assert receipt.terminal_status == "failed_closed"
    assert receipt.mcts_turns == ()


def test_local_command_builder_has_no_factory_and_therefore_fails_closed(tmp_path) -> None:
    schedule = build_bo1000_schedule(_digest("command-builder-schedule"))
    envelope = build_bo1000_pair_envelope(
        schedule,
        pair_index=1,
        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
        pair_rng_snapshot_sha256=_digest("command-builder-rng"),
        deck_order_rng_sha256=_digest("command-builder-deck"),
        execution_host="command-builder-host",
    )
    controller = HostLocalBO1000PairController(
        tmp_path / "out", execution_host="command-builder-host"
    )

    from poke_bot.recursive_turn_planner.bo1000_pair_runner import HostLocalSubprocessGameRunner

    result = controller.run_pair(envelope, HostLocalSubprocessGameRunner(build_r207_local_game_command))
    assert result.status.status == "failed_closed"
    assert all(receipt.crash_count == 1 for receipt in result.game_receipts)


def test_completed_local_receipt_compiles_into_the_exact_bo1000_report() -> None:
    request = _request()
    receipt = R207LocalGameCommandAdapter(_live_factory([])).execute(request)
    assert receipt.terminal_status == "completed"
    schedule = build_bo1000_schedule(_digest("receipt-compiler-schedule"))
    receipts: list[BO1000GameReceipt] = []
    for spec in schedule:
        turn = replace(
            receipt.mcts_turns[0],
            game_nonce_sha256=spec.game_nonce_sha256,
            pair_id=spec.pair_id,
            mcts_seat=spec.mcts_seat,
            planner_turn_id=f"receipt-{spec.pair_index}-{spec.game_index}",
            turn_key=(spec.mcts_seat, spec.pair_index),
        )
        receipts.append(
            BO1000GameReceipt(
                game_nonce_sha256=spec.game_nonce_sha256,
                pair_id=spec.pair_id,
                game_index=spec.game_index,
                mcts_seat=spec.mcts_seat,
                no_rtp_seat=spec.no_rtp_seat,
                pair_rng_snapshot_sha256=_digest(f"report-rng:{spec.pair_index}"),
                deck_order_rng_sha256=_digest(f"report-deck:{spec.pair_index}"),
                checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
                bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
                terminal_status="completed",
                winner_seat=0,
                illegal_action_count=0,
                forfeit_count=0,
                crash_count=0,
                timeout_count=0,
                mcts_turns=(turn,),
            )
        )
    report = compile_bo1000_report(
        schedule,
        receipts,
        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
    )
    assert report["identities"]["evaluation_id"] == (
        "alakazam-r207-simulator-backed-chance-aware-inter-turn-mcts-bo1000"
    )
    assert report["support"]["scheduled_games"] == 1000
