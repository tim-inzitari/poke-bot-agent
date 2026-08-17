"""Strict schedule and report compiler for the r207 BO1000 shadow study.

This module does not run games, load a model, or grant action authority.  It
owns the deterministic 500-pair/1,000-game schedule shape and validates the
immutable game/turn receipts produced by a separately preflighted simulator
MCTS runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

BO1000_PAIR_COUNT = 500
BO1000_GAME_COUNT = 1_000
R207_EVALUATION_ID = (
    "alakazam-r207-simulator-backed-chance-aware-inter-turn-mcts-bo1000"
)
MCTS_ARM = "simulator_backed_chance_aware_inter_turn_mcts"
CONTROL_ARM = "no_rtp_direct_policy"
BO1000_REPORT_SCHEMA = "poke_bot.alakazam_chance_aware_mcts_bo1000_r207_report/v1"
# These are deliberately constants rather than caller-selected report knobs.
# They bind the BO1000 compiler to the exact r207 typed contract currently
# staged by GOAL.md. A different budget or contract needs a new evaluation
# identity and compiler/receipt path; it cannot be relabelled at report time.
R207_CONTRACT_SHA256 = (
    "sha256:d9cb5f8d15e2bebbcbf943f5a273a4116703c3e8549a3328b7d78d161f7b5dce"
)
# The exact schema whose canonical payload produced the fixed digest below.
# r207's arena-native persistent session emits this digest, not the legacy
# SimulatorOneTurnExpectimax configuration identity.
R207_CANONICAL_PLANNER_CONFIG_SCHEMA = "poke_bot.chance_aware_search_config/v1"
R207_CANONICAL_PLANNER_CONFIG_SHA256 = (
    "sha256:95fcbccd6b74b757c9baf14fc55ce7984a2abc13d37ee9552e313789fe9d3560"
)
R207_MAX_TURN_SECONDS = 20.0
R207_MAX_ACTION_SECONDS = 5.0
R207_FROZEN_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R207_FROZEN_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)


class BO1000EvidenceError(ValueError):
    """Raised when scheduled or observed BO1000 evidence is inconsistent."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise BO1000EvidenceError(f"{name} must be a sha256 digest")
    suffix = value.removeprefix("sha256:")
    if len(suffix) != 64 or any(ch not in "0123456789abcdef" for ch in suffix):
        raise BO1000EvidenceError(f"{name} must be a lowercase sha256 digest")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BO1000EvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, minimum: float = 0.0) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise BO1000EvidenceError(f"{name} must be finite")
    result = float(value)
    if result < minimum:
        raise BO1000EvidenceError(f"{name} must be >= {minimum}")
    return result


def _expected_game_nonce(pair_nonce_sha256: str, game_index: int) -> str:
    return _canonical_sha256(
        {
            "schema": "poke_bot.alakazam_mcts_bo1000_game_request/v1",
            "pair_nonce_sha256": pair_nonce_sha256,
            "game_index": game_index,
        }
    )


@dataclass(frozen=True, slots=True)
class BO1000GameSpec:
    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    game_index: int
    game_nonce_sha256: str
    mcts_seat: int
    no_rtp_seat: int

    def __post_init__(self) -> None:
        _exact_int(self.pair_index, name="pair_index")
        if not self.pair_id:
            raise BO1000EvidenceError("pair_id must be nonempty")
        _digest(self.pair_nonce_sha256, name="pair_nonce_sha256")
        if self.game_index not in {0, 1}:
            raise BO1000EvidenceError("game_index must be 0 or 1")
        _digest(self.game_nonce_sha256, name="game_nonce_sha256")
        if self.game_nonce_sha256 != _expected_game_nonce(
            self.pair_nonce_sha256, self.game_index
        ):
            raise BO1000EvidenceError(
                "game_nonce_sha256 must bind pair nonce and game index"
            )
        if self.mcts_seat not in {0, 1} or self.no_rtp_seat != 1 - self.mcts_seat:
            raise BO1000EvidenceError("game seats must be exact opposites")

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def build_bo1000_schedule(seed_identity_sha256: str) -> tuple[BO1000GameSpec, ...]:
    """Build the exact deterministic 500-pair, seat-swapped request schedule."""

    _digest(seed_identity_sha256, name="seed_identity_sha256")
    games: list[BO1000GameSpec] = []
    for pair_index in range(BO1000_PAIR_COUNT):
        pair_nonce = _canonical_sha256(
            {
                "schema": "poke_bot.alakazam_mcts_bo1000_pair_request/v1",
                "seed_identity_sha256": seed_identity_sha256,
                "pair_index": pair_index,
            }
        )
        pair_id = f"r207-pair-{pair_index:06d}-{pair_nonce[7:19]}"
        for game_index in (0, 1):
            game_nonce = _expected_game_nonce(pair_nonce, game_index)
            games.append(
                BO1000GameSpec(
                    pair_index=pair_index,
                    pair_id=pair_id,
                    pair_nonce_sha256=pair_nonce,
                    game_index=game_index,
                    game_nonce_sha256=game_nonce,
                    mcts_seat=game_index,
                    no_rtp_seat=1 - game_index,
                )
            )
    return tuple(games)


@dataclass(frozen=True, slots=True)
class MCTSTurnTelemetry:
    game_nonce_sha256: str
    pair_id: str
    mcts_seat: int
    planner_turn_id: str
    turn_key: tuple[int, int]
    actions_dispatched: int
    simulator_transitions_seen: int
    result_or_leaf_evaluations_seen: int
    simulator_leaf_evaluations_seen: int
    neural_leaf_evaluations_seen: int
    unique_tree_nodes_seen: int
    decision_nodes_expanded: int
    terminal_exact_results_seen: int
    boundary_leaf_results_seen: int
    finite_chance_outcomes_evaluated: int
    frozen_policy_prior_batches: int
    frozen_policy_prior_evaluations: int
    batched_frozen_outcome_value_leaf_reranking_batches: int
    frozen_outcome_leaf_evaluations: int
    frozen_value_leaf_evaluations: int
    nonterminal_leaves_reranked: bool
    terminal_exact_results_not_reranked: bool
    cache_hits: int
    deterministic_subtree_reuses: int
    tree_rebuilds: int
    turn_planner_wall_seconds: float
    max_single_action_planner_wall_seconds: float
    requested_tree_fully_expanded_and_backed_up_within_budget: bool
    tree_incomplete_reason: str | None
    deadline_hit: bool
    direct_fallback_used: bool
    selected_action_legal: bool
    selected_action_sha256: str
    legal_actions_sha256: str
    tree_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        _digest(self.game_nonce_sha256, name="turn game_nonce_sha256")
        if not self.pair_id or not self.planner_turn_id:
            raise BO1000EvidenceError("turn pair_id/planner_turn_id must be nonempty")
        if self.mcts_seat not in {0, 1}:
            raise BO1000EvidenceError("turn mcts_seat must be 0 or 1")
        if (
            not isinstance(self.turn_key, tuple)
            or len(self.turn_key) != 2
            or any(type(v) is not int or v < 0 for v in self.turn_key)
        ):
            raise BO1000EvidenceError("turn_key must be two nonnegative integers")
        for field in (
            "actions_dispatched",
            "simulator_transitions_seen",
            "result_or_leaf_evaluations_seen",
            "simulator_leaf_evaluations_seen",
            "neural_leaf_evaluations_seen",
            "unique_tree_nodes_seen",
            "decision_nodes_expanded",
            "terminal_exact_results_seen",
            "boundary_leaf_results_seen",
            "finite_chance_outcomes_evaluated",
            "frozen_policy_prior_batches",
            "frozen_policy_prior_evaluations",
            "batched_frozen_outcome_value_leaf_reranking_batches",
            "frozen_outcome_leaf_evaluations",
            "frozen_value_leaf_evaluations",
            "cache_hits",
            "deterministic_subtree_reuses",
            "tree_rebuilds",
        ):
            _exact_int(getattr(self, field), name=field)
        if self.actions_dispatched != 1:
            raise BO1000EvidenceError(
                "every MCTS turn must dispatch exactly one atomic action"
            )
        if self.result_or_leaf_evaluations_seen != (
            self.terminal_exact_results_seen + self.neural_leaf_evaluations_seen
        ):
            raise BO1000EvidenceError(
                "result_or_leaf_evaluations_seen must equal terminal exact + neural leaves"
            )
        if self.terminal_exact_results_seen != self.simulator_leaf_evaluations_seen:
            raise BO1000EvidenceError(
                "simulator leaf evaluations must be exact terminal results"
            )
        if self.boundary_leaf_results_seen > self.neural_leaf_evaluations_seen:
            raise BO1000EvidenceError(
                "boundary leaf results cannot exceed neural leaf evaluations"
            )
        if self.simulator_transitions_seen < (
            self.terminal_exact_results_seen + self.boundary_leaf_results_seen
        ):
            raise BO1000EvidenceError(
                "simulator transitions cannot be fewer than terminal and boundary results"
            )
        if self.frozen_outcome_leaf_evaluations != self.neural_leaf_evaluations_seen:
            raise BO1000EvidenceError("outcome-head leaf count mismatch")
        if self.frozen_value_leaf_evaluations != self.neural_leaf_evaluations_seen:
            raise BO1000EvidenceError("value-head leaf count mismatch")
        if self.frozen_policy_prior_evaluations < self.decision_nodes_expanded:
            raise BO1000EvidenceError(
                "policy priors must cover every expanded decision node"
            )
        if (
            self.frozen_policy_prior_evaluations > 0
            and self.frozen_policy_prior_batches == 0
        ):
            raise BO1000EvidenceError("policy-prior evaluations require a batch")
        if (
            self.frozen_policy_prior_batches > 0
            and self.frozen_policy_prior_evaluations == 0
        ):
            raise BO1000EvidenceError("policy-prior batches must contain evaluations")
        if self.frozen_policy_prior_batches > self.frozen_policy_prior_evaluations:
            raise BO1000EvidenceError("policy-prior batches cannot exceed evaluations")
        if (
            self.neural_leaf_evaluations_seen > 0
            and self.batched_frozen_outcome_value_leaf_reranking_batches == 0
        ):
            raise BO1000EvidenceError("neural leaf evaluations require a batch")
        if (
            self.batched_frozen_outcome_value_leaf_reranking_batches > 0
            and self.neural_leaf_evaluations_seen == 0
        ):
            raise BO1000EvidenceError(
                "outcome/value leaf batches must contain evaluations"
            )
        if (
            self.batched_frozen_outcome_value_leaf_reranking_batches
            > self.neural_leaf_evaluations_seen
        ):
            raise BO1000EvidenceError(
                "outcome/value leaf batches cannot exceed evaluations"
            )
        if type(self.nonterminal_leaves_reranked) is not bool:
            raise BO1000EvidenceError("nonterminal_leaves_reranked must be boolean")
        if type(self.terminal_exact_results_not_reranked) is not bool:
            raise BO1000EvidenceError(
                "terminal_exact_results_not_reranked must be boolean"
            )
        if self.nonterminal_leaves_reranked != bool(self.neural_leaf_evaluations_seen):
            raise BO1000EvidenceError(
                "nonterminal reranking flag must match neural leaf evaluation count"
            )
        if not self.terminal_exact_results_not_reranked:
            raise BO1000EvidenceError("exact terminal results may never be reranked")
        turn_seconds = _finite(
            self.turn_planner_wall_seconds,
            name="turn_planner_wall_seconds",
        )
        action_seconds = _finite(
            self.max_single_action_planner_wall_seconds,
            name="max_single_action_planner_wall_seconds",
        )
        if turn_seconds > R207_MAX_TURN_SECONDS:
            raise BO1000EvidenceError(
                "turn planner time exceeds exact r207 20.0-second budget"
            )
        if action_seconds > R207_MAX_ACTION_SECONDS:
            raise BO1000EvidenceError(
                "single-action planner time exceeds exact r207 5.0-second budget"
            )
        if action_seconds > turn_seconds:
            raise BO1000EvidenceError(
                "single-action planner time cannot exceed total turn planner time"
            )
        if (
            type(self.requested_tree_fully_expanded_and_backed_up_within_budget)
            is not bool
        ):
            raise BO1000EvidenceError("full-tree status must be boolean")
        if self.requested_tree_fully_expanded_and_backed_up_within_budget:
            if (
                self.tree_incomplete_reason is not None
                or self.deadline_hit
                or self.direct_fallback_used
            ):
                raise BO1000EvidenceError(
                    "a complete tree cannot carry an incomplete reason, deadline hit, or fallback"
                )
        else:
            if not self.tree_incomplete_reason:
                raise BO1000EvidenceError("an incomplete tree needs an exact reason")
            if not self.direct_fallback_used:
                raise BO1000EvidenceError(
                    "an incomplete tree must use the exact direct fallback"
                )
        for field in (
            "deadline_hit",
            "direct_fallback_used",
            "selected_action_legal",
        ):
            if type(getattr(self, field)) is not bool:
                raise BO1000EvidenceError(f"{field} must be boolean")
        if not self.selected_action_legal:
            raise BO1000EvidenceError("MCTS selected action must be exactly legal")
        for field in (
            "selected_action_sha256",
            "legal_actions_sha256",
            "tree_sha256",
            "config_sha256",
        ):
            _digest(getattr(self, field), name=field)
        if self.config_sha256 != R207_CANONICAL_PLANNER_CONFIG_SHA256:
            raise BO1000EvidenceError(
                "turn config_sha256 is not the canonical r207 20.0s/5.0s planner config"
            )

    @property
    def terminal_results_seen(self) -> int:
        """Read-only legacy alias; serialized r207 evidence uses the exact name."""

        return self.terminal_exact_results_seen


@dataclass(frozen=True, slots=True)
class BO1000GameReceipt:
    game_nonce_sha256: str
    pair_id: str
    game_index: int
    mcts_seat: int
    no_rtp_seat: int
    pair_rng_snapshot_sha256: str
    deck_order_rng_sha256: str
    checkpoint_sha256: str
    bundle_sha256: str
    terminal_status: str
    winner_seat: int | None
    illegal_action_count: int
    forfeit_count: int
    crash_count: int
    timeout_count: int
    mcts_turns: tuple[MCTSTurnTelemetry, ...]

    def __post_init__(self) -> None:
        _digest(self.game_nonce_sha256, name="game_nonce_sha256")
        if not self.pair_id:
            raise BO1000EvidenceError("pair_id must be nonempty")
        if self.game_index not in {0, 1}:
            raise BO1000EvidenceError("game_index must be 0 or 1")
        if self.mcts_seat not in {0, 1} or self.no_rtp_seat != 1 - self.mcts_seat:
            raise BO1000EvidenceError("receipt seats must be exact opposites")
        for field in (
            "pair_rng_snapshot_sha256",
            "deck_order_rng_sha256",
            "checkpoint_sha256",
            "bundle_sha256",
        ):
            _digest(getattr(self, field), name=field)
        if self.terminal_status not in {"completed", "failed_closed"}:
            raise BO1000EvidenceError(
                "terminal_status must be completed or failed_closed"
            )
        if self.winner_seat not in {None, 0, 1}:
            raise BO1000EvidenceError(
                "winner_seat must be 0, 1, or None for draw/failure"
            )
        if self.terminal_status == "failed_closed" and self.winner_seat is not None:
            raise BO1000EvidenceError("failed-closed game cannot claim a winner")
        for field in (
            "illegal_action_count",
            "forfeit_count",
            "crash_count",
            "timeout_count",
        ):
            _exact_int(getattr(self, field), name=field)
        if not isinstance(self.mcts_turns, tuple) or any(
            not isinstance(turn, MCTSTurnTelemetry) for turn in self.mcts_turns
        ):
            raise BO1000EvidenceError("mcts_turns must be a tuple of MCTSTurnTelemetry")
        seen_turns: set[str] = set()
        seen_turn_keys: set[tuple[int, int]] = set()
        for turn in self.mcts_turns:
            if turn.game_nonce_sha256 != self.game_nonce_sha256:
                raise BO1000EvidenceError("turn/game nonce binding mismatch")
            if turn.pair_id != self.pair_id or turn.mcts_seat != self.mcts_seat:
                raise BO1000EvidenceError("turn/game pair or seat binding mismatch")
            if turn.planner_turn_id in seen_turns:
                raise BO1000EvidenceError("duplicate planner_turn_id in game")
            seen_turns.add(turn.planner_turn_id)
            if turn.turn_key in seen_turn_keys:
                raise BO1000EvidenceError("duplicate turn_key in game")
            seen_turn_keys.add(turn.turn_key)


def _distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "minimum": None,
            "maximum": None,
        }
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    p95 = ordered[max(0, math.ceil(0.95 * n) - 1)]
    p99 = ordered[max(0, math.ceil(0.99 * n) - 1)]
    return {
        "count": n,
        "mean": sum(ordered) / n,
        "median": statistics.median(ordered),
        "p95": p95,
        "p99": p99,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _outcome(receipt: BO1000GameReceipt) -> str:
    if receipt.terminal_status != "completed":
        return "failed_closed"
    if receipt.winner_seat is None:
        return "draw"
    return "win" if receipt.winner_seat == receipt.mcts_seat else "loss"


def _control_outcome(receipt: BO1000GameReceipt) -> str:
    if receipt.terminal_status != "completed":
        return "failed_closed"
    if receipt.winner_seat is None:
        return "draw"
    return "win" if receipt.winner_seat == receipt.no_rtp_seat else "loss"


def _outcome_counts(outcomes: Iterable[str]) -> dict[str, int]:
    materialized = tuple(outcomes)
    return {
        name: materialized.count(name)
        for name in ("win", "draw", "loss", "failed_closed")
    }


def _validate_schedule(
    schedule: Sequence[BO1000GameSpec],
) -> tuple[dict[str, BO1000GameSpec], dict[str, list[BO1000GameSpec]]]:
    if len(schedule) != BO1000_GAME_COUNT:
        raise BO1000EvidenceError("schedule must contain exactly 1000 games")
    by_nonce: dict[str, BO1000GameSpec] = {}
    pair_specs: dict[str, list[BO1000GameSpec]] = {}
    for spec in schedule:
        if not isinstance(spec, BO1000GameSpec):
            raise BO1000EvidenceError("schedule must contain BO1000GameSpec values")
        if spec.game_nonce_sha256 in by_nonce:
            raise BO1000EvidenceError("duplicate game nonce in schedule")
        by_nonce[spec.game_nonce_sha256] = spec
        pair_specs.setdefault(spec.pair_id, []).append(spec)
    if len(pair_specs) != BO1000_PAIR_COUNT:
        raise BO1000EvidenceError("schedule must contain exactly 500 unique pairs")
    if {spec.pair_index for spec in schedule} != set(range(BO1000_PAIR_COUNT)):
        raise BO1000EvidenceError("schedule pair indices must be exactly 0 through 499")

    pair_nonce_to_id: dict[str, str] = {}
    for pair_id, specs in pair_specs.items():
        if len(specs) != 2:
            raise BO1000EvidenceError("every pair must contain exactly two games")
        if sorted((spec.game_index, spec.mcts_seat) for spec in specs) != [
            (0, 0),
            (1, 1),
        ]:
            raise BO1000EvidenceError(
                "every pair must contain exact seat-swapped games"
            )
        if len({spec.pair_index for spec in specs}) != 1:
            raise BO1000EvidenceError("pair index mismatch inside scheduled pair")
        pair_nonces = {spec.pair_nonce_sha256 for spec in specs}
        if len(pair_nonces) != 1:
            raise BO1000EvidenceError("pair request nonce mismatch")
        pair_nonce = next(iter(pair_nonces))
        expected_pair_id = f"r207-pair-{specs[0].pair_index:06d}-{pair_nonce.removeprefix('sha256:')[:12]}"
        if pair_id != expected_pair_id:
            raise BO1000EvidenceError("pair_id must bind the r207 pair index and nonce")
        existing_pair_id = pair_nonce_to_id.setdefault(pair_nonce, pair_id)
        if existing_pair_id != pair_id:
            raise BO1000EvidenceError("pair request nonce is reused across pairs")
    return by_nonce, pair_specs


_TURN_COMPONENT_FIELDS = (
    "result_or_leaf_evaluations_seen",
    "simulator_transitions_seen",
    "terminal_exact_results_seen",
    "boundary_leaf_results_seen",
    "simulator_leaf_evaluations_seen",
    "neural_leaf_evaluations_seen",
    "frozen_policy_prior_batches",
    "frozen_policy_prior_evaluations",
    "batched_frozen_outcome_value_leaf_reranking_batches",
    "frozen_outcome_leaf_evaluations",
    "frozen_value_leaf_evaluations",
    "unique_tree_nodes_seen",
    "decision_nodes_expanded",
)


def _turn_component_distributions(
    turns: Sequence[MCTSTurnTelemetry],
) -> dict[str, dict[str, float | int | None]]:
    return {
        f"{field}_per_turn": _distribution([getattr(turn, field) for turn in turns])
        for field in _TURN_COMPONENT_FIELDS
    }


def _turn_reranking_integrity(turns: Sequence[MCTSTurnTelemetry]) -> dict[str, int]:
    return {
        "turns_with_nonterminal_leaves_reranked": sum(
            turn.nonterminal_leaves_reranked for turn in turns
        ),
        "turns_with_terminal_exact_results_not_reranked": sum(
            turn.terminal_exact_results_not_reranked for turn in turns
        ),
    }


def _validate_r207_turn_contract(turn: MCTSTurnTelemetry) -> None:
    """Recheck immutable turn facts before compiling a terminal report.

    Dataclass construction already validates these properties, but report
    compilation treats receipts as untrusted evidence and repeats the critical
    identity and timing checks. This prevents a post-construction mutation from
    turning a timing breach into a benign report counter.
    """

    # ``frozen=True`` is not a security boundary: callers can still construct
    # a receipt and mutate it with ``object.__setattr__`` before it reaches the
    # report compiler. Reconstruct the typed record so finite timings, exact
    # budgets, config identity, and every component-count invariant are checked
    # again rather than treating a bad receipt as a reportable breach.
    try:
        MCTSTurnTelemetry(
            **{
                field: getattr(turn, field)
                for field in MCTSTurnTelemetry.__dataclass_fields__
            }
        )
    except (BO1000EvidenceError, TypeError) as exc:
        raise BO1000EvidenceError(
            "turn receipt violates the canonical r207 MCTS contract"
        ) from exc


def _has_material_mcts_telemetry(turn: MCTSTurnTelemetry) -> bool:
    """Return whether one turn proves real simulator-backed MCTS work occurred."""

    return (
        turn.actions_dispatched == 1
        and turn.simulator_transitions_seen > 0
        and turn.decision_nodes_expanded > 0
        and turn.frozen_policy_prior_batches > 0
        and turn.frozen_policy_prior_evaluations > 0
        and turn.result_or_leaf_evaluations_seen > 0
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def compile_bo1000_report(
    schedule: Sequence[BO1000GameSpec],
    receipts: Iterable[BO1000GameReceipt],
    *,
    checkpoint_sha256: str,
    bundle_sha256: str,
    max_turn_seconds: float | None = None,
    max_action_seconds: float | None = None,
) -> dict[str, object]:
    """Validate all pair/game/turn evidence and compile the terminal report."""

    _digest(checkpoint_sha256, name="checkpoint_sha256")
    _digest(bundle_sha256, name="bundle_sha256")
    for value, expected, name in (
        (max_turn_seconds, R207_MAX_TURN_SECONDS, "max_turn_seconds"),
        (max_action_seconds, R207_MAX_ACTION_SECONDS, "max_action_seconds"),
    ):
        if value is not None and _finite(value, name=name) != expected:
            raise BO1000EvidenceError(
                f"{name} is fixed by the canonical r207 contract and cannot be overridden"
            )
    by_nonce, _pair_specs = _validate_schedule(schedule)

    observed: dict[str, BO1000GameReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, BO1000GameReceipt):
            raise BO1000EvidenceError("receipts must contain BO1000GameReceipt values")
        if receipt.game_nonce_sha256 in observed:
            raise BO1000EvidenceError("duplicate game receipt")
        spec = by_nonce.get(receipt.game_nonce_sha256)
        if spec is None:
            raise BO1000EvidenceError("receipt is not in the immutable schedule")
        if (
            receipt.pair_id != spec.pair_id
            or receipt.game_index != spec.game_index
            or receipt.mcts_seat != spec.mcts_seat
            or receipt.no_rtp_seat != spec.no_rtp_seat
        ):
            raise BO1000EvidenceError("receipt schedule identity mismatch")
        if receipt.checkpoint_sha256 != checkpoint_sha256:
            raise BO1000EvidenceError("same-checkpoint invariant failed")
        if receipt.bundle_sha256 != bundle_sha256:
            raise BO1000EvidenceError("same-bundle invariant failed")
        observed[receipt.game_nonce_sha256] = receipt
    missing = sorted(set(by_nonce) - set(observed))
    if missing:
        raise BO1000EvidenceError(f"missing {len(missing)} scheduled game receipts")

    pair_receipts: dict[str, list[BO1000GameReceipt]] = {}
    for receipt in observed.values():
        pair_receipts.setdefault(receipt.pair_id, []).append(receipt)
    for pair_id, pair in pair_receipts.items():
        if len(pair) != 2:
            raise BO1000EvidenceError(f"pair {pair_id} must have exactly two receipts")
        if len({game.pair_rng_snapshot_sha256 for game in pair}) != 1:
            raise BO1000EvidenceError(f"pair {pair_id} RNG snapshot mismatch")
        if len({game.deck_order_rng_sha256 for game in pair}) != 1:
            raise BO1000EvidenceError(f"pair {pair_id} deck-order RNG mismatch")

    games = [observed[spec.game_nonce_sha256] for spec in schedule]
    for game in games:
        # Failed-closed games may legitimately contain no turn, but any turn
        # they do preserve must still be valid evidence. Otherwise a malformed
        # timing/config/count receipt could contaminate the aggregate report.
        for turn in game.mcts_turns:
            _validate_r207_turn_contract(turn)
        if game.terminal_status != "completed":
            continue
        if not game.mcts_turns:
            raise BO1000EvidenceError(
                "completed experimental MCTS game is missing MCTS turn telemetry"
            )
        if not any(_has_material_mcts_telemetry(turn) for turn in game.mcts_turns):
            raise BO1000EvidenceError(
                "completed experimental MCTS game has no material MCTS telemetry"
            )
    outcomes = [_outcome(game) for game in games]
    outcome_counts = _outcome_counts(outcomes)
    control_outcomes = [_control_outcome(game) for game in games]
    control_outcome_counts = _outcome_counts(control_outcomes)
    seat_outcomes: dict[str, dict[str, int]] = {}
    for seat in (0, 1):
        selected = [_outcome(game) for game in games if game.mcts_seat == seat]
        seat_outcomes[str(seat)] = _outcome_counts(selected)

    pair_matrix: dict[str, int] = {}
    pair_scores: list[float] = []
    failed_closed_pair_ids: list[str] = []
    for pair_id in sorted(pair_receipts):
        pair = sorted(pair_receipts[pair_id], key=lambda game: game.game_index)
        labels = tuple(_outcome(game) for game in pair)
        pair_matrix[f"{labels[0]}__{labels[1]}"] = (
            pair_matrix.get(f"{labels[0]}__{labels[1]}", 0) + 1
        )
        if "failed_closed" not in labels:
            score = (
                sum(
                    1.0 if label == "win" else 0.5 if label == "draw" else 0.0
                    for label in labels
                )
                / 2.0
            )
            pair_scores.append(score)
        else:
            failed_closed_pair_ids.append(pair_id)
    pair_mean = None if not pair_scores else sum(pair_scores) / len(pair_scores)
    paired_difference = None if pair_mean is None else pair_mean - 0.5
    if len(pair_scores) >= 2:
        standard_error = statistics.stdev(pair_scores) / math.sqrt(len(pair_scores))
        paired_ci = [
            paired_difference - 1.96 * standard_error,
            paired_difference + 1.96 * standard_error,
        ]  # type: ignore[operator]
    else:
        paired_ci = None

    turns = [turn for game in games for turn in game.mcts_turns]
    complete_turns = [
        turn
        for turn in turns
        if turn.requested_tree_fully_expanded_and_backed_up_within_budget
    ]
    incomplete_turns = [
        turn
        for turn in turns
        if not turn.requested_tree_fully_expanded_and_backed_up_within_budget
    ]
    component_distributions = _turn_component_distributions(turns)
    result_counts = [turn.result_or_leaf_evaluations_seen for turn in turns]
    result_counts_by_seat = {
        str(seat): _distribution(
            [
                turn.result_or_leaf_evaluations_seen
                for turn in turns
                if turn.mcts_seat == seat
            ]
        )
        for seat in (0, 1)
    }
    result_counts_by_completion = {
        "complete": _distribution(
            [turn.result_or_leaf_evaluations_seen for turn in complete_turns]
        ),
        "incomplete": _distribution(
            [turn.result_or_leaf_evaluations_seen for turn in incomplete_turns]
        ),
    }
    component_splits = {
        "by_mcts_seat": {
            str(seat): _turn_component_distributions(
                [turn for turn in turns if turn.mcts_seat == seat]
            )
            for seat in (0, 1)
        },
        "by_tree_completion": {
            "complete": _turn_component_distributions(complete_turns),
            "incomplete": _turn_component_distributions(incomplete_turns),
        },
        "terminal_exact_vs_nonterminal_frozen_leaf": {
            "terminal_exact_results_seen_per_turn": component_distributions[
                "terminal_exact_results_seen_per_turn"
            ],
            "boundary_leaf_results_seen_per_turn": component_distributions[
                "boundary_leaf_results_seen_per_turn"
            ],
            "nonterminal_frozen_leaf_evaluations_seen_per_turn": component_distributions[
                "neural_leaf_evaluations_seen_per_turn"
            ],
            **_turn_reranking_integrity(turns),
        },
        "policy_prior_outcome_leaf_value_leaf": {
            key: component_distributions[key]
            for key in (
                "frozen_policy_prior_batches_per_turn",
                "frozen_policy_prior_evaluations_per_turn",
                "batched_frozen_outcome_value_leaf_reranking_batches_per_turn",
                "frozen_outcome_leaf_evaluations_per_turn",
                "frozen_value_leaf_evaluations_per_turn",
            )
        },
    }
    turn_breaches = sum(
        turn.turn_planner_wall_seconds > R207_MAX_TURN_SECONDS for turn in turns
    )
    action_breaches = sum(
        turn.max_single_action_planner_wall_seconds > R207_MAX_ACTION_SECONDS
        for turn in turns
    )
    if turn_breaches or action_breaches:
        raise BO1000EvidenceError(
            "r207 timing breach must fail closed, not be reported"
        )
    incomplete_reasons: dict[str, int] = {}
    for turn in incomplete_turns:
        assert turn.tree_incomplete_reason is not None
        incomplete_reasons[turn.tree_incomplete_reason] = (
            incomplete_reasons.get(turn.tree_incomplete_reason, 0) + 1
        )

    report: dict[str, object] = {
        "schema": BO1000_REPORT_SCHEMA,
        "status": "complete_with_runtime_failures"
        if outcome_counts["failed_closed"]
        else "complete",
        "support": {
            "scheduled_games": len(schedule),
            "observed_terminal_game_receipts": len(games),
            "rng_matched_pairs": len(pair_receipts),
            "mcts_as_seat_0": sum(game.mcts_seat == 0 for game in games),
            "mcts_as_seat_1": sum(game.mcts_seat == 1 for game in games),
            "no_rtp_as_seat_0": sum(game.no_rtp_seat == 0 for game in games),
            "no_rtp_as_seat_1": sum(game.no_rtp_seat == 1 for game in games),
            "mcts_turns": len(turns),
            "paired_analysis_eligible_pairs": len(pair_scores),
            "paired_analysis_excluded_failed_closed_pairs": len(failed_closed_pair_ids),
        },
        "identities": {
            "evaluation_id": R207_EVALUATION_ID,
            "experimental_arm": MCTS_ARM,
            "control_arm": CONTROL_ARM,
            "r207_contract_sha256": R207_CONTRACT_SHA256,
            "canonical_planner_config_schema": R207_CANONICAL_PLANNER_CONFIG_SCHEMA,
            "canonical_planner_config_sha256": R207_CANONICAL_PLANNER_CONFIG_SHA256,
            "checkpoint_sha256": checkpoint_sha256,
            "bundle_sha256": bundle_sha256,
            "schedule_sha256": _canonical_sha256(
                [spec.as_payload() for spec in schedule]
            ),
        },
        "game_outcomes": {
            "by_arm": {
                MCTS_ARM: outcome_counts,
                CONTROL_ARM: control_outcome_counts,
            },
            "mcts_perspective": outcome_counts,
            "by_mcts_seat": seat_outcomes,
            "paired_outcome_matrix": dict(sorted(pair_matrix.items())),
            "paired_analysis": {
                "eligible_completed_pairs": len(pair_scores),
                "excluded_failed_closed_pairs": len(failed_closed_pair_ids),
                "excluded_failed_closed_pair_ids_sha256": _canonical_sha256(
                    failed_closed_pair_ids
                ),
                "imputation_used": False,
            },
            "paired_mcts_score": pair_mean,
            "paired_win_rate_difference": paired_difference,
            "paired_confidence_interval": paired_ci,
            "paired_score_difference_from_0_5": paired_difference,
            "paired_normal_95_percent_ci": paired_ci,
            "paired_ci_method": "pair-clustered normal approximation over two-game mean scores",
            "illegal_action_count": sum(game.illegal_action_count for game in games),
            "forfeit_count": sum(game.forfeit_count for game in games),
            "crash_count": sum(game.crash_count for game in games),
            "timeout_count": sum(game.timeout_count for game in games),
        },
        "search_throughput": {
            "result_or_leaf_evaluations_seen_per_turn": _distribution(result_counts),
            "simulator_transitions_seen_per_turn": _distribution(
                [turn.simulator_transitions_seen for turn in turns]
            ),
            "terminal_exact_results_seen_per_turn": _distribution(
                [turn.terminal_exact_results_seen for turn in turns]
            ),
            "boundary_leaf_results_seen_per_turn": _distribution(
                [turn.boundary_leaf_results_seen for turn in turns]
            ),
            "simulator_leaf_evaluations_seen_per_turn": _distribution(
                [turn.simulator_leaf_evaluations_seen for turn in turns]
            ),
            "neural_leaf_evaluations_seen_per_turn": _distribution(
                [turn.neural_leaf_evaluations_seen for turn in turns]
            ),
            "frozen_policy_prior_batches_per_turn": _distribution(
                [turn.frozen_policy_prior_batches for turn in turns]
            ),
            "frozen_policy_prior_evaluations_per_turn": _distribution(
                [turn.frozen_policy_prior_evaluations for turn in turns]
            ),
            "frozen_outcome_value_leaf_batches_per_turn": _distribution(
                [
                    turn.batched_frozen_outcome_value_leaf_reranking_batches
                    for turn in turns
                ]
            ),
            "frozen_outcome_leaf_evaluations_per_turn": _distribution(
                [turn.frozen_outcome_leaf_evaluations for turn in turns]
            ),
            "frozen_value_leaf_evaluations_per_turn": _distribution(
                [turn.frozen_value_leaf_evaluations for turn in turns]
            ),
            "by_mcts_seat": result_counts_by_seat,
            "by_tree_completion": result_counts_by_completion,
            "component_splits": component_splits,
            "unique_tree_nodes_seen_per_turn": _distribution(
                [turn.unique_tree_nodes_seen for turn in turns]
            ),
            "decision_nodes_expanded_per_turn": _distribution(
                [turn.decision_nodes_expanded for turn in turns]
            ),
        },
        "tree_completion": {
            "complete_turns": len(complete_turns),
            "incomplete_turns": len(incomplete_turns),
            "full_tree_completion_rate": _rate(len(complete_turns), len(turns)),
            "full_tree_completion_rate_by_mcts_seat": {
                str(seat): _rate(
                    sum(
                        turn.requested_tree_fully_expanded_and_backed_up_within_budget
                        for turn in turns
                        if turn.mcts_seat == seat
                    ),
                    sum(turn.mcts_seat == seat for turn in turns),
                )
                for seat in (0, 1)
            },
            "incomplete_reasons": dict(sorted(incomplete_reasons.items())),
            "deadline_hit_turns": sum(turn.deadline_hit for turn in turns),
            "direct_fallback_turns": sum(turn.direct_fallback_used for turn in turns),
            "deterministic_subtree_reuses": sum(
                turn.deterministic_subtree_reuses for turn in turns
            ),
            "tree_rebuilds": sum(turn.tree_rebuilds for turn in turns),
            "cache_hits": sum(turn.cache_hits for turn in turns),
            "finite_chance_outcomes_evaluated": sum(
                turn.finite_chance_outcomes_evaluated for turn in turns
            ),
        },
        "latency": {
            "planner_time_per_turn_seconds": _distribution(
                [turn.turn_planner_wall_seconds for turn in turns]
            ),
            "maximum_planner_time_per_action_seconds": _distribution(
                [turn.max_single_action_planner_wall_seconds for turn in turns]
            ),
            "turn_budget_seconds": R207_MAX_TURN_SECONDS,
            "action_budget_seconds": R207_MAX_ACTION_SECONDS,
            "turn_budget_breach_count": turn_breaches,
            "action_budget_breach_count": action_breaches,
            "direct_fallback_after_deadline_count": sum(
                turn.deadline_hit and turn.direct_fallback_used for turn in turns
            ),
        },
        "authority": {
            "training_eligible": False,
            "serving_eligible": False,
            "selector_change_authorized": False,
            "promotion_authorized": False,
        },
        "telemetry_integrity": {
            "raw_per_game_and_per_turn_receipts_preserved": True,
            "missing_telemetry_may_be_imputed": False,
            "paired_failed_closed_outcomes_imputed": False,
            "completed_experimental_games_require_material_mcts_telemetry": True,
            "result_or_leaf_count_invariant": (
                "terminal_exact_results_seen + neural_leaf_evaluations_seen"
            ),
            "r207_timing_override_allowed": False,
        },
    }
    report["canonical_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "BO1000_GAME_COUNT",
    "BO1000_PAIR_COUNT",
    "BO1000_REPORT_SCHEMA",
    "CONTROL_ARM",
    "MCTS_ARM",
    "R207_CANONICAL_PLANNER_CONFIG_SCHEMA",
    "R207_CANONICAL_PLANNER_CONFIG_SHA256",
    "R207_CONTRACT_SHA256",
    "R207_EVALUATION_ID",
    "R207_MAX_ACTION_SECONDS",
    "R207_MAX_TURN_SECONDS",
    "BO1000EvidenceError",
    "BO1000GameReceipt",
    "BO1000GameSpec",
    "MCTSTurnTelemetry",
    "build_bo1000_schedule",
    "compile_bo1000_report",
]
