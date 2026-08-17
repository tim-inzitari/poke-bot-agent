"""Batched, frozen-policy leaf scoring for the r205 simulator search.

This is deliberately an *evaluation component*, not a simulator, tree builder,
or action bridge.  The one-turn MCTS owns successor generation, exact chance
handling, backup, cache reuse, and the final full-tree-completion claim.  This
module only does four narrowly scoped things:

* preserves every simulator-provided complete action and returns frozen-model
  policy priors in that exact order;
* bypasses the neural network for an exact terminal simulator result;
* batches non-terminal, policy-visible leaves through the frozen r195 model;
* makes an uncalibrated value/outcome score diagnostic-only, so it cannot be
  mistaken for a receipted search value or action override.

The r206 guide-logit A/B experiment is intentionally absent here.  It is a
separate serving-package experiment and must not leak into the r205 NO-RTP
mirror arm or into the neural MCTS leaf evaluator.

No checkpoint is written, no gradients are computed, no simulator state is
owned, and no result from this module has production action authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # Keep the generic scorer importable without Torch.
    from poke_bot.model import TemporalCabtTransformer


Action = tuple[int, ...]

R205_NO_RTP_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R205_NO_RTP_CHECKPOINT_BYTES = 127_914_385
R205_NO_RTP_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R205_ALAKAZAM_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R205_MAX_COMPLETE_ACTIONS = 1024
R205_ABSOLUTE_NEURAL_BATCH_ROWS = 1024

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class NeuralLeafRerankerError(ValueError):
    """A leaf, identity, score receipt, or batched forward is invalid."""


class LeafKind(str, Enum):
    """The simulator-derived state represented by one requested leaf."""

    TERMINAL = "terminal"
    SUCCESSOR = "successor"
    BOUNDARY = "boundary"


def _require_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NeuralLeafRerankerError(f"{label} must be a canonical sha256 digest")
    return value


def _require_finite(value: float, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise NeuralLeafRerankerError(f"{label} must be finite")
    return float(value)


def _normalise_action(value: Action, *, label: str) -> Action:
    if not isinstance(value, tuple):
        raise NeuralLeafRerankerError(f"{label} must be a tuple")
    if any(type(part) is not int or part < 0 for part in value):
        raise NeuralLeafRerankerError(
            f"{label} members must be non-negative exact integers"
        )
    return value


def _normalise_actions(
    value: tuple[Action, ...],
    *,
    label: str,
    maximum: int,
) -> tuple[Action, ...]:
    if not isinstance(value, tuple) or not value:
        raise NeuralLeafRerankerError(f"{label} must be a non-empty tuple")
    result = tuple(
        _normalise_action(action, label=f"{label}[{index}]")
        for index, action in enumerate(value)
    )
    if len(result) > maximum:
        raise NeuralLeafRerankerError(
            f"{label} exceeds the complete-action ceiling of {maximum}"
        )
    if len(set(result)) != len(result):
        raise NeuralLeafRerankerError(f"{label} must not contain duplicates")
    return result


def _canonical_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NeuralLeafRerankerError("identity payload is not canonical JSON") from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _complete_action_fingerprint(actions: tuple[Action, ...]) -> str:
    return _canonical_sha256(
        {
            "schema": "poke_bot.complete_ordered_action_fingerprint/v1",
            "actions": [list(action) for action in actions],
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenPolicyIdentity:
    """Byte identities that must be shared by both r205 evaluation arms."""

    checkpoint_sha256: str
    checkpoint_bytes: int
    bundle_sha256: str
    deck_cards_sha256: str
    nonplanner_runtime_sha256: str

    def __post_init__(self) -> None:
        _require_digest(self.checkpoint_sha256, label="checkpoint_sha256")
        _require_digest(self.bundle_sha256, label="bundle_sha256")
        _require_digest(self.deck_cards_sha256, label="deck_cards_sha256")
        _require_digest(
            self.nonplanner_runtime_sha256,
            label="nonplanner_runtime_sha256",
        )
        if type(self.checkpoint_bytes) is not int or self.checkpoint_bytes <= 0:
            raise NeuralLeafRerankerError("checkpoint_bytes must be a positive exact int")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "poke_bot.r205_frozen_policy_identity/v1",
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_bytes": self.checkpoint_bytes,
                "bundle_sha256": self.bundle_sha256,
                "deck_cards_sha256": self.deck_cards_sha256,
                "nonplanner_runtime_sha256": self.nonplanner_runtime_sha256,
            }
        )

    @classmethod
    def r205_no_rtp(cls) -> FrozenPolicyIdentity:
        """Return the exact r195 NO-RTP identity required by r205."""

        return cls(
            checkpoint_sha256=R205_NO_RTP_CHECKPOINT_SHA256,
            checkpoint_bytes=R205_NO_RTP_CHECKPOINT_BYTES,
            bundle_sha256=R205_NO_RTP_BUNDLE_SHA256,
            deck_cards_sha256=R205_ALAKAZAM_DECK_CARDS_SHA256,
            # The submitted NO-RTP bundle is the immutable non-planner runtime.
            nonplanner_runtime_sha256=R205_NO_RTP_BUNDLE_SHA256,
        )


@dataclass(frozen=True, slots=True)
class LeafScoreCalibration:
    """A reference to a separately verified non-terminal score receipt.

    This object deliberately cannot assert that an external receipt exists.
    The r205 launch preflight must independently open and verify that immutable
    receipt.  It does bind score weights and model identity so a result cannot
    be reused under another checkpoint, deck/runtime, or blend.
    """

    receipt_sha256: str
    frozen_policy_identity_sha256: str
    value_weight: float
    outcome_weight: float
    value_head_calibrated: bool
    outcome_distribution_calibrated: bool
    source_excluded: bool
    policy_visible_only: bool
    evaluation_only: bool = True
    training_eligible: bool = False
    serving_action_authority: bool = False
    outcome_class_order: tuple[str, str, str] = ("loss", "draw", "win")

    def __post_init__(self) -> None:
        _require_digest(self.receipt_sha256, label="calibration receipt")
        _require_digest(
            self.frozen_policy_identity_sha256,
            label="calibration frozen policy identity",
        )
        value_weight = _require_finite(self.value_weight, label="value_weight")
        outcome_weight = _require_finite(
            self.outcome_weight,
            label="outcome_weight",
        )
        if value_weight < 0.0 or outcome_weight < 0.0:
            raise NeuralLeafRerankerError("leaf score weights must be non-negative")
        if not math.isclose(value_weight + outcome_weight, 1.0, abs_tol=1e-12):
            raise NeuralLeafRerankerError("leaf score weights must sum exactly to 1")
        if value_weight > 0.0 and self.value_head_calibrated is not True:
            raise NeuralLeafRerankerError(
                "a nonzero value score requires value-head calibration"
            )
        if (
            outcome_weight > 0.0
            and self.outcome_distribution_calibrated is not True
        ):
            raise NeuralLeafRerankerError(
                "a nonzero outcome score requires outcome calibration"
            )
        for label in (
            "value_head_calibrated",
            "outcome_distribution_calibrated",
            "source_excluded",
            "policy_visible_only",
            "evaluation_only",
            "training_eligible",
            "serving_action_authority",
        ):
            if type(getattr(self, label)) is not bool:
                raise NeuralLeafRerankerError(f"{label} must be an exact bool")
        if self.source_excluded is not True or self.policy_visible_only is not True:
            raise NeuralLeafRerankerError(
                "non-terminal leaf calibration must be source-excluded and policy-visible"
            )
        if self.evaluation_only is not True or self.training_eligible is not False:
            raise NeuralLeafRerankerError(
                "leaf calibration must remain evaluation-only and training-ineligible"
            )
        if self.serving_action_authority is not False:
            raise NeuralLeafRerankerError(
                "leaf calibration cannot grant serving action authority"
            )
        if self.outcome_class_order != ("loss", "draw", "win"):
            raise NeuralLeafRerankerError(
                "outcome head class order must be exactly (loss, draw, win)"
            )


@dataclass(frozen=True, slots=True)
class LeafDeadline:
    """Absolute monotonic deadlines supplied by the one typed search budget."""

    turn_deadline_monotonic: float
    action_deadline_monotonic: float

    def __post_init__(self) -> None:
        _require_finite(
            self.turn_deadline_monotonic,
            label="turn_deadline_monotonic",
        )
        _require_finite(
            self.action_deadline_monotonic,
            label="action_deadline_monotonic",
        )

    @property
    def expires_at_monotonic(self) -> float:
        return min(self.turn_deadline_monotonic, self.action_deadline_monotonic)

    def expired(self, now: float) -> bool:
        return _require_finite(now, label="monotonic clock") >= self.expires_at_monotonic

    def remaining_seconds(self, now: float) -> float:
        return max(
            0.0,
            self.expires_at_monotonic - _require_finite(now, label="monotonic clock"),
        )


@dataclass(frozen=True, slots=True)
class NeuralLeafRerankerConfig:
    """Stable, evaluation-only batch/scoring configuration.

    Time ceilings are intentionally not duplicated here.  The owning
    :class:`ChanceAwareSearchConfig` supplies the sole typed 20 s / 5 s budget
    and creates :class:`LeafDeadline` for each call.
    """

    frozen_policy_identity: FrozenPolicyIdentity
    max_microbatch_leaves: int = 64
    max_complete_actions: int = R205_MAX_COMPLETE_ACTIONS
    require_outcome_distribution: bool = True
    calibration: LeafScoreCalibration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_policy_identity, FrozenPolicyIdentity):
            raise NeuralLeafRerankerError("frozen_policy_identity is required")
        if (
            type(self.max_microbatch_leaves) is not int
            or not 1 <= self.max_microbatch_leaves <= R205_ABSOLUTE_NEURAL_BATCH_ROWS
        ):
            raise NeuralLeafRerankerError(
                "max_microbatch_leaves must be an exact int in [1, 1024]"
            )
        if (
            type(self.max_complete_actions) is not int
            or not 1 <= self.max_complete_actions <= R205_MAX_COMPLETE_ACTIONS
        ):
            raise NeuralLeafRerankerError(
                "max_complete_actions must be an exact int in [1, 1024]"
            )
        if type(self.require_outcome_distribution) is not bool:
            raise NeuralLeafRerankerError(
                "require_outcome_distribution must be an exact bool"
            )
        calibration = self.calibration
        if calibration is not None:
            if not isinstance(calibration, LeafScoreCalibration):
                raise NeuralLeafRerankerError("calibration has an invalid type")
            if (
                calibration.frozen_policy_identity_sha256
                != self.frozen_policy_identity.identity_sha256
            ):
                raise NeuralLeafRerankerError(
                    "calibration belongs to a different frozen policy identity"
                )

    @property
    def identity_sha256(self) -> str:
        calibration = self.calibration
        return _canonical_sha256(
            {
                "schema": "poke_bot.r205_neural_leaf_reranker_config/v1",
                "frozen_policy_identity_sha256": (
                    self.frozen_policy_identity.identity_sha256
                ),
                "max_microbatch_leaves": self.max_microbatch_leaves,
                "max_complete_actions": self.max_complete_actions,
                "require_outcome_distribution": self.require_outcome_distribution,
                "calibration": (
                    None
                    if calibration is None
                    else {
                        "receipt_sha256": calibration.receipt_sha256,
                        "value_weight": calibration.value_weight,
                        "outcome_weight": calibration.outcome_weight,
                    }
                ),
                "guide_logit_bonus_used": False,
                "production_action_authority": False,
            }
        )


@dataclass(frozen=True, slots=True)
class LeafRequest:
    """One simulator-produced leaf; non-terminals must retain exact legality."""

    request_id: str
    kind: LeafKind
    simulator_result_sha256: str
    public_state_sha256: str
    root_seat: int
    packet: object | None = None
    expected_actions: tuple[Action, ...] = ()
    direct_action: Action | None = None
    terminal_value: Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise NeuralLeafRerankerError("request_id must be a non-empty string")
        if not isinstance(self.kind, LeafKind):
            raise NeuralLeafRerankerError("kind must be a LeafKind")
        _require_digest(self.simulator_result_sha256, label="simulator_result_sha256")
        _require_digest(self.public_state_sha256, label="public_state_sha256")
        if type(self.root_seat) is not int or self.root_seat not in (0, 1):
            raise NeuralLeafRerankerError("root_seat must be exactly 0 or 1")

        if self.kind is LeafKind.TERMINAL:
            if self.packet is not None:
                raise NeuralLeafRerankerError("terminal leaves must not carry a network packet")
            if self.expected_actions or self.direct_action is not None:
                raise NeuralLeafRerankerError(
                    "terminal leaves must not carry legal actions or a direct action"
                )
            if not isinstance(self.terminal_value, Fraction):
                raise NeuralLeafRerankerError("terminal_value must be an exact Fraction")
            if self.terminal_value < -1 or self.terminal_value > 1:
                raise NeuralLeafRerankerError("terminal_value must be in [-1, 1]")
            return

        if self.packet is None:
            raise NeuralLeafRerankerError("non-terminal leaves require a policy packet")
        actions = _normalise_actions(
            self.expected_actions,
            label="expected_actions",
            maximum=R205_MAX_COMPLETE_ACTIONS,
        )
        object.__setattr__(self, "expected_actions", actions)
        if self.direct_action is None:
            raise NeuralLeafRerankerError("non-terminal leaves require direct_action")
        direct = _normalise_action(self.direct_action, label="direct_action")
        object.__setattr__(self, "direct_action", direct)
        if direct not in actions:
            raise NeuralLeafRerankerError(
                "the exact direct-policy action must be in expected_actions"
            )
        if self.terminal_value is not None:
            raise NeuralLeafRerankerError(
                "non-terminal leaves must not claim an exact terminal value"
            )

    @property
    def action_fingerprint_sha256(self) -> str | None:
        if self.kind is LeafKind.TERMINAL:
            return None
        return _complete_action_fingerprint(self.expected_actions)


@dataclass(frozen=True, slots=True)
class NeuralForwardRow:
    """Raw model output for one non-terminal leaf in request order."""

    policy_logits: tuple[float, ...]
    value_from_actor: float
    acting_seat: int
    outcome_distribution_logits_from_actor: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_logits, tuple) or not self.policy_logits:
            raise NeuralLeafRerankerError("policy_logits must be a non-empty tuple")
        for index, value in enumerate(self.policy_logits):
            _require_finite(value, label=f"policy_logits[{index}]")
        _require_finite(self.value_from_actor, label="value_from_actor")
        if type(self.acting_seat) is not int or self.acting_seat not in (0, 1):
            raise NeuralLeafRerankerError("acting_seat must be exactly 0 or 1")
        logits = self.outcome_distribution_logits_from_actor
        if logits is not None:
            if not isinstance(logits, tuple) or len(logits) != 3:
                raise NeuralLeafRerankerError(
                    "outcome_distribution_logits_from_actor must have three logits"
                )
            for index, value in enumerate(logits):
                _require_finite(value, label=f"outcome logits[{index}]")


class NeuralLeafBatchBackend(Protocol):
    """A pure ordered batch forward over policy-visible non-terminal leaves."""

    frozen_policy_identity_sha256: str

    def __call__(self, requests: Sequence[LeafRequest]) -> Sequence[NeuralForwardRow]:
        """Return exactly one raw row for each request, in the same order."""


@dataclass(frozen=True, slots=True)
class RootPolicyPriors:
    """Exact complete-action priors; stable order doubles as tie-break order."""

    actions: tuple[Action, ...]
    probabilities: tuple[float, ...]
    raw_logits: tuple[float, ...]
    direct_action_index: int
    ranked_action_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LeafEvaluation:
    """One terminal exact result or one frozen-model non-terminal result."""

    request_id: str
    kind: LeafKind
    simulator_result_sha256: str
    public_state_sha256: str
    exact_terminal_value: Fraction | None
    root_policy_priors: RootPolicyPriors | None
    raw_value_from_root: float | None
    outcome_probabilities_from_root: tuple[float, float, float] | None
    raw_outcome_expected_value_from_root: float | None
    selected_leaf_value: float | None
    value_source: str
    search_value_eligible: bool


@dataclass(frozen=True, slots=True)
class LeafRerankerTelemetry:
    """Leaf-batch facts; tree completion remains the MCTS builder's claim."""

    simulator_results_seen: int
    simulator_terminal_results_seen: int
    simulator_boundary_leaves_seen: int
    neural_evaluations_started: int
    neural_evaluations_accepted: int
    neural_forward_batches: int
    root_policy_prior_evaluations: int
    outcome_head_evaluations: int
    frozen_policy_prior_batches: int
    frozen_policy_prior_evaluations: int
    batched_frozen_outcome_value_leaf_reranking_batches: int
    frozen_outcome_leaf_evaluations: int
    frozen_value_leaf_evaluations: int
    nonterminal_leaves_reranked: int
    terminal_exact_results_not_reranked: int
    result_or_leaf_evaluations_seen: int
    exact_terminal_values_used: int
    requested_leaf_batch_completed: bool
    every_leaf_has_search_eligible_value: bool
    deadline_hit: bool
    incomplete_reason: str | None
    elapsed_seconds: float
    full_tree_completion_owned_by: str = "simulator_mcts_tree_builder"


@dataclass(frozen=True, slots=True)
class LeafRerankerResult:
    evaluations: tuple[LeafEvaluation, ...]
    telemetry: LeafRerankerTelemetry


def _stable_softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(logits)
    terms = tuple(math.exp(value - maximum) for value in logits)
    total = sum(terms)
    if not math.isfinite(total) or total <= 0.0:
        raise NeuralLeafRerankerError("policy softmax is not finite")
    return tuple(term / total for term in terms)


def _outcome_probabilities(logits: tuple[float, float, float]) -> tuple[float, float, float]:
    probabilities = _stable_softmax(logits)
    return (probabilities[0], probabilities[1], probabilities[2])


class BatchedNeuralLeafReranker:
    """Run a fail-closed ordered batch through a frozen neural backend.

    The class never selects an action.  A caller may use ``root_policy_priors``
    for PUCT/progressive expansion only after the surrounding MCTS launch fence
    has validated its own simulator, legality, chance, and calibration receipts.
    """

    production_action_authority_enabled = False
    guide_logit_bonus_used = False

    def __init__(
        self,
        backend: NeuralLeafBatchBackend,
        config: NeuralLeafRerankerConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not callable(backend):
            raise NeuralLeafRerankerError("backend must be callable")
        if not isinstance(config, NeuralLeafRerankerConfig):
            raise NeuralLeafRerankerError("config must be NeuralLeafRerankerConfig")
        backend_identity = getattr(backend, "frozen_policy_identity_sha256", None)
        _require_digest(
            backend_identity,
            label="backend frozen policy identity",
        )
        if backend_identity != config.frozen_policy_identity.identity_sha256:
            raise NeuralLeafRerankerError(
                "backend belongs to a different frozen policy identity"
            )
        self._backend = backend
        self.config = config
        self._clock = clock or time.monotonic
        if not callable(self._clock):
            raise NeuralLeafRerankerError("clock must be callable")

    def _now(self) -> float:
        return _require_finite(self._clock(), label="monotonic clock")

    def evaluate(
        self,
        requests: Sequence[LeafRequest],
        *,
        deadline: LeafDeadline,
    ) -> LeafRerankerResult:
        """Score one ordered batch or discard it entirely at its deadline.

        A completed first microbatch is intentionally not returned if a later
        microbatch times out or fails.  The owning planner must then discard its
        partial tree and use its exact direct-policy fallback as r202 requires.
        """

        if not isinstance(deadline, LeafDeadline):
            raise NeuralLeafRerankerError("deadline must be LeafDeadline")
        rows = tuple(requests)
        if not rows:
            raise NeuralLeafRerankerError("a leaf batch must not be empty")
        if any(not isinstance(row, LeafRequest) for row in rows):
            raise NeuralLeafRerankerError("every batch member must be LeafRequest")
        ids = tuple(row.request_id for row in rows)
        if len(set(ids)) != len(ids):
            raise NeuralLeafRerankerError("leaf request_id values must be unique")
        if any(
            len(row.expected_actions) > self.config.max_complete_actions
            for row in rows
            if row.kind is not LeafKind.TERMINAL
        ):
            raise NeuralLeafRerankerError("complete-action cap exceeded")

        started_at = self._now()
        terminals = tuple(row for row in rows if row.kind is LeafKind.TERMINAL)
        nonterminals = tuple(row for row in rows if row.kind is not LeafKind.TERMINAL)
        base = {
            "simulator_results_seen": len(rows),
            "simulator_terminal_results_seen": len(terminals),
            "simulator_boundary_leaves_seen": sum(
                row.kind is LeafKind.BOUNDARY for row in rows
            ),
            "exact_terminal_values_used": len(terminals),
        }

        if deadline.expired(started_at):
            return self._incomplete(
                started_at=started_at,
                base=base,
                started=0,
                batches=0,
                reason="deadline_before_neural_leaf_batch",
            )

        evaluations: dict[str, LeafEvaluation] = {
            row.request_id: LeafEvaluation(
                request_id=row.request_id,
                kind=row.kind,
                simulator_result_sha256=row.simulator_result_sha256,
                public_state_sha256=row.public_state_sha256,
                exact_terminal_value=row.terminal_value,
                root_policy_priors=None,
                raw_value_from_root=None,
                outcome_probabilities_from_root=None,
                raw_outcome_expected_value_from_root=None,
                selected_leaf_value=float(row.terminal_value),
                value_source="exact_simulator_terminal",
                search_value_eligible=True,
            )
            for row in terminals
        }

        started = 0
        batches = 0
        accepted = 0
        priors_seen = 0
        outcomes_seen = 0
        for offset in range(0, len(nonterminals), self.config.max_microbatch_leaves):
            chunk = nonterminals[offset : offset + self.config.max_microbatch_leaves]
            if deadline.expired(self._now()):
                return self._incomplete(
                    started_at=started_at,
                    base=base,
                    started=started,
                    batches=batches,
                    reason="deadline_before_neural_microbatch",
                )
            try:
                raw_rows = tuple(self._backend(chunk))
            except (
                NeuralLeafRerankerError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                # The outer planner records its typed backend error.
                return self._incomplete(
                    started_at=started_at,
                    base=base,
                    started=started + len(chunk),
                    batches=batches + 1,
                    reason="neural_backend_failure",
                )
            started += len(chunk)
            batches += 1
            if len(raw_rows) != len(chunk):
                return self._incomplete(
                    started_at=started_at,
                    base=base,
                    started=started,
                    batches=batches,
                    reason="neural_backend_row_count_mismatch",
                )
            if deadline.expired(self._now()):
                return self._incomplete(
                    started_at=started_at,
                    base=base,
                    started=started,
                    batches=batches,
                    reason="deadline_after_neural_microbatch",
                )
            try:
                for request, raw in zip(chunk, raw_rows):
                    evaluation = self._build_nonterminal_evaluation(request, raw)
                    evaluations[request.request_id] = evaluation
                    accepted += 1
                    priors_seen += 1
                    outcomes_seen += int(
                        raw.outcome_distribution_logits_from_actor is not None
                    )
                    if deadline.expired(self._now()):
                        return self._incomplete(
                            started_at=started_at,
                            base=base,
                            started=started,
                            batches=batches,
                            reason="deadline_during_neural_output_validation",
                        )
            except NeuralLeafRerankerError:
                return self._incomplete(
                    started_at=started_at,
                    base=base,
                    started=started,
                    batches=batches,
                    reason="invalid_neural_leaf_output",
                )

        if deadline.expired(self._now()):
            return self._incomplete(
                started_at=started_at,
                base=base,
                started=started,
                batches=batches,
                reason="deadline_after_neural_leaf_validation",
            )
        elapsed = self._now() - started_at
        if elapsed < 0.0:
            raise NeuralLeafRerankerError("monotonic clock moved backwards")
        ordered = tuple(evaluations[row.request_id] for row in rows)
        every_eligible = all(row.search_value_eligible for row in ordered)
        return LeafRerankerResult(
            evaluations=ordered,
            telemetry=LeafRerankerTelemetry(
                **base,
                neural_evaluations_started=started,
                neural_evaluations_accepted=accepted,
                neural_forward_batches=batches,
                root_policy_prior_evaluations=priors_seen,
                outcome_head_evaluations=outcomes_seen,
                frozen_policy_prior_batches=batches,
                frozen_policy_prior_evaluations=priors_seen,
                batched_frozen_outcome_value_leaf_reranking_batches=batches,
                frozen_outcome_leaf_evaluations=outcomes_seen,
                frozen_value_leaf_evaluations=accepted,
                nonterminal_leaves_reranked=accepted,
                terminal_exact_results_not_reranked=len(terminals),
                result_or_leaf_evaluations_seen=(
                    base["simulator_results_seen"] + accepted
                ),
                requested_leaf_batch_completed=True,
                every_leaf_has_search_eligible_value=every_eligible,
                deadline_hit=False,
                incomplete_reason=(
                    None if every_eligible else "uncalibrated_nonterminal_leaf_value"
                ),
                elapsed_seconds=elapsed,
            ),
        )

    def _incomplete(
        self,
        *,
        started_at: float,
        base: dict[str, int],
        started: int,
        batches: int,
        reason: str,
    ) -> LeafRerankerResult:
        elapsed = self._now() - started_at
        if elapsed < 0.0:
            raise NeuralLeafRerankerError("monotonic clock moved backwards")
        return LeafRerankerResult(
            evaluations=(),
            telemetry=LeafRerankerTelemetry(
                **base,
                neural_evaluations_started=started,
                neural_evaluations_accepted=0,
                neural_forward_batches=batches,
                root_policy_prior_evaluations=0,
                outcome_head_evaluations=0,
                frozen_policy_prior_batches=batches,
                frozen_policy_prior_evaluations=0,
                batched_frozen_outcome_value_leaf_reranking_batches=batches,
                frozen_outcome_leaf_evaluations=0,
                frozen_value_leaf_evaluations=0,
                nonterminal_leaves_reranked=0,
                terminal_exact_results_not_reranked=base[
                    "simulator_terminal_results_seen"
                ],
                result_or_leaf_evaluations_seen=base["simulator_results_seen"],
                requested_leaf_batch_completed=False,
                every_leaf_has_search_eligible_value=False,
                deadline_hit=reason.startswith("deadline_"),
                incomplete_reason=reason,
                elapsed_seconds=elapsed,
            ),
        )

    def _build_nonterminal_evaluation(
        self,
        request: LeafRequest,
        raw: NeuralForwardRow,
    ) -> LeafEvaluation:
        if not isinstance(raw, NeuralForwardRow):
            raise NeuralLeafRerankerError("backend must return NeuralForwardRow")
        if len(raw.policy_logits) != len(request.expected_actions):
            raise NeuralLeafRerankerError(
                "neural policy length does not match exact complete legal actions"
            )
        priors = _stable_softmax(raw.policy_logits)
        direct_action = request.direct_action
        assert direct_action is not None  # enforced by LeafRequest
        direct_index = request.expected_actions.index(direct_action)
        # Stable canonical order breaks ties; do not let a Python hash alter it.
        ranked = tuple(
            sorted(
                range(len(priors)),
                key=lambda index: (-priors[index], index),
            )
        )
        root_sign = 1.0 if raw.acting_seat == request.root_seat else -1.0
        root_value = root_sign * float(raw.value_from_actor)
        if root_value < -1.000001 or root_value > 1.000001:
            raise NeuralLeafRerankerError("value head must be in the game-value range")

        outcome_probabilities: tuple[float, float, float] | None = None
        outcome_expected: float | None = None
        if raw.outcome_distribution_logits_from_actor is not None:
            actor_probabilities = _outcome_probabilities(
                raw.outcome_distribution_logits_from_actor
            )
            # [loss, draw, win] is actor-relative; swap loss/win for root.
            outcome_probabilities = (
                actor_probabilities
                if root_sign > 0.0
                else (
                    actor_probabilities[2],
                    actor_probabilities[1],
                    actor_probabilities[0],
                )
            )
            outcome_expected = outcome_probabilities[2] - outcome_probabilities[0]
        elif self.config.require_outcome_distribution:
            raise NeuralLeafRerankerError(
                "r207 non-terminal leaves require frozen outcome-distribution logits"
            )

        calibration = self.config.calibration
        selected_value: float | None = None
        source = "untrusted_network_diagnostic"
        eligible = False
        if calibration is not None:
            score = 0.0
            if calibration.value_weight:
                score += calibration.value_weight * root_value
            if calibration.outcome_weight:
                if outcome_expected is None:
                    raise NeuralLeafRerankerError(
                        "calibrated outcome score requested but outcome head is unavailable"
                    )
                score += calibration.outcome_weight * outcome_expected
            selected_value = score
            eligible = True
            source = (
                "calibrated_value_head"
                if calibration.outcome_weight == 0.0
                else (
                    "calibrated_outcome_distribution"
                    if calibration.value_weight == 0.0
                    else "calibrated_value_outcome_blend"
                )
            )

        return LeafEvaluation(
            request_id=request.request_id,
            kind=request.kind,
            simulator_result_sha256=request.simulator_result_sha256,
            public_state_sha256=request.public_state_sha256,
            exact_terminal_value=None,
            root_policy_priors=RootPolicyPriors(
                actions=request.expected_actions,
                probabilities=priors,
                raw_logits=raw.policy_logits,
                direct_action_index=direct_index,
                ranked_action_indices=ranked,
            ),
            raw_value_from_root=root_value,
            outcome_probabilities_from_root=outcome_probabilities,
            raw_outcome_expected_value_from_root=outcome_expected,
            selected_leaf_value=selected_value,
            value_source=source,
            search_value_eligible=eligible,
        )


class TemporalModelLeafBackend:
    """Torch adapter for the frozen Temporal model, with no search-side cache.

    Construct this only around a separate evaluation model that is already in
    ``eval()`` mode and whose parameters are frozen.  The adapter performs one
    batch forward with ``kv_cache=None`` / ``append_cache=False`` (or the
    stateless history batch equivalent), checks RNG purity, and never imports
    a simulator search API.
    """

    def __init__(
        self,
        model: TemporalCabtTransformer,
        *,
        frozen_policy_identity: FrozenPolicyIdentity,
    ) -> None:
        # Avoid importing Torch merely to import the generic scorer.
        import torch

        if not isinstance(frozen_policy_identity, FrozenPolicyIdentity):
            raise NeuralLeafRerankerError(
                "frozen_policy_identity must be FrozenPolicyIdentity"
            )
        if model.training:
            raise NeuralLeafRerankerError("frozen model must already be in eval mode")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise NeuralLeafRerankerError(
                "frozen model parameters must have requires_grad=False"
            )
        self._model = model
        self._torch = torch
        self.frozen_policy_identity_sha256 = frozen_policy_identity.identity_sha256

    def __call__(self, requests: Sequence[LeafRequest]) -> Sequence[NeuralForwardRow]:
        from poke_bot.batched_infer import featurize_packets

        if not requests:
            return ()
        if any(request.kind is LeafKind.TERMINAL for request in requests):
            raise NeuralLeafRerankerError("terminal leaves must bypass the neural backend")
        packets = []
        for request in requests:
            packet = request.packet
            if packet is None:
                raise NeuralLeafRerankerError("non-terminal request lost its packet")
            if getattr(packet, "root_seat", None) != request.root_seat:
                raise NeuralLeafRerankerError("packet root seat does not match request")
            packets.append(packet)

        torch = self._torch
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = self._cuda_rng_state()
        try:
            features = featurize_packets(packets)
            for request, combos in zip(requests, features.combos):
                observed_actions = tuple(tuple(int(part) for part in combo) for combo in combos)
                if observed_actions != request.expected_actions:
                    raise NeuralLeafRerankerError(
                        "packet action enumeration differs from simulator-bound legality"
                    )
            with torch.no_grad():
                if getattr(self._model, "decision_context", "stateless") == "history":
                    output = self._model.forward_history_batch(
                        features.histories,
                        features.opts,
                        n_options=features.n_opts,
                        previous_action_histories=features.previous_action_histories,
                        matchup_routes=features.matchup_routes,
                    )
                else:
                    output = self._model.forward(
                        features.boards,
                        features.opts,
                        kv_cache=None,
                        append_cache=False,
                        n_options=features.n_opts,
                        matchup_routes=features.matchup_routes,
                    )
                outcome_logits = None
                if bool(getattr(self._model, "expanded_heads_enabled", False)):
                    state_vec = output.get("state_vec")
                    if state_vec is None:
                        raise NeuralLeafRerankerError("model forward omitted state_vec")
                    outcome_logits = self._model.expanded_state_logits(state_vec).get(
                        "outcome_distribution"
                    )

                policy_logits = output.get("policy_logits")
                values = output.get("value")
                if policy_logits is None or values is None:
                    raise NeuralLeafRerankerError("model forward omitted policy/value")
                policy_cpu = policy_logits.detach().float().cpu()
                value_cpu = values.detach().float().cpu()
                outcome_cpu = (
                    None
                    if outcome_logits is None
                    else outcome_logits.detach().float().cpu()
                )
        finally:
            if not torch.equal(cpu_rng_before, torch.random.get_rng_state()):
                raise NeuralLeafRerankerError("neural leaf forward mutated CPU RNG state")
            cuda_rng_after = self._cuda_rng_state()
            if (
                cuda_rng_before is not None
                and cuda_rng_after is not None
                and not torch.equal(cuda_rng_before, cuda_rng_after)
            ):
                raise NeuralLeafRerankerError(
                    "neural leaf forward mutated CUDA RNG state"
                )

        rows: list[NeuralForwardRow] = []
        for index, request in enumerate(requests):
            n_actions = len(request.expected_actions)
            outcomes = (
                None
                if outcome_cpu is None
                else tuple(float(value) for value in outcome_cpu[index].tolist())
            )
            rows.append(
                NeuralForwardRow(
                    policy_logits=tuple(
                        float(value) for value in policy_cpu[index, :n_actions].tolist()
                    ),
                    value_from_actor=float(value_cpu[index].item()),
                    acting_seat=int(features.seats[index]),
                    outcome_distribution_logits_from_actor=outcomes,
                )
            )
        return tuple(rows)

    def _cuda_rng_state(self):
        torch = self._torch
        try:
            device = next(self._model.parameters()).device
        except StopIteration as exc:
            raise NeuralLeafRerankerError("model has no parameters") from exc
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        return torch.cuda.get_rng_state(device).clone()


def verify_frozen_checkpoint_bytes(
    path: str | Path,
    identity: FrozenPolicyIdentity,
) -> Path:
    """Verify the checkpoint byte identity before loading an isolated model."""

    if not isinstance(identity, FrozenPolicyIdentity):
        raise NeuralLeafRerankerError("identity must be FrozenPolicyIdentity")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise NeuralLeafRerankerError("frozen checkpoint path is not a regular file")
    observed_size = resolved.stat().st_size
    if observed_size != identity.checkpoint_bytes:
        raise NeuralLeafRerankerError("frozen checkpoint byte count mismatch")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    observed = f"sha256:{digest.hexdigest()}"
    if observed != identity.checkpoint_sha256:
        raise NeuralLeafRerankerError("frozen checkpoint sha256 mismatch")
    return resolved


def load_isolated_frozen_temporal_backend(
    checkpoint_path: str | Path,
    *,
    identity: FrozenPolicyIdentity,
    device: object | None = None,
) -> TemporalModelLeafBackend:
    """Load a separately owned frozen model after byte-level identity proof.

    This helper does not reload, patch, or touch any managed model service.  It
    is suitable only for a newly staged r205 evaluation worker after the remote
    capacity/noninterference preflight has succeeded.
    """

    verified = verify_frozen_checkpoint_bytes(checkpoint_path, identity)
    import torch

    from poke_bot.train import load_model_from_checkpoint

    selected_device = device if device is not None else torch.device("cpu")
    model = load_model_from_checkpoint(verified, device=selected_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return TemporalModelLeafBackend(model, frozen_policy_identity=identity)


__all__ = [
    "R205_ALAKAZAM_DECK_CARDS_SHA256",
    "R205_MAX_COMPLETE_ACTIONS",
    "R205_NO_RTP_BUNDLE_SHA256",
    "R205_NO_RTP_CHECKPOINT_BYTES",
    "R205_NO_RTP_CHECKPOINT_SHA256",
    "Action",
    "BatchedNeuralLeafReranker",
    "FrozenPolicyIdentity",
    "LeafDeadline",
    "LeafEvaluation",
    "LeafKind",
    "LeafRequest",
    "LeafRerankerResult",
    "LeafRerankerTelemetry",
    "LeafScoreCalibration",
    "NeuralForwardRow",
    "NeuralLeafBatchBackend",
    "NeuralLeafRerankerConfig",
    "NeuralLeafRerankerError",
    "RootPolicyPriors",
    "TemporalModelLeafBackend",
    "load_isolated_frozen_temporal_backend",
    "verify_frozen_checkpoint_bytes",
]
