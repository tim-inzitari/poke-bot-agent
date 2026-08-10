"""Evaluation-only simulator successor facility for the r207 MCTS preflight.

The installed r198 pairing-snapshot ABI is deliberately *not* a mid-game clone
API.  This module therefore does not amend or reinterpret it.  Instead it
offers a truthful replay arena: each simulated edge restores the sealed initial
snapshot into a fresh scratch battle and replays an auditable public action
journal before making one scratch selection.

That is simulator-backed work, but it is intentionally conservative:

* no native state bytes, seed, private zone, or raw battle pointer leaves the
  arena;
* a v2 replay can advance only a transition explicitly attested as public and
  deterministic;
* chance, opponent, private-information, malformed, and unattested paths are
  boundary leaves.  In particular, v2 cannot enumerate or force chance and
  must never call a sampled outcome an exact expectimax child;
* a future v3 native implementation can implement :class:`SuccessorArena` and
  return opaque finite-chance children with exact ``Fraction`` probabilities;
* every operation is checked against one absolute 20-second turn / 5-second
  action deadline.  A result returning after a deadline is discarded.

The module deliberately imports neither a policy/model nor legacy MCTS code.
Frozen r195 policy-prior and leaf-reranker callers are injected at a higher
evaluation layer only after their separate identity and no-mutation receipts.
Nothing here has serving or action authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Protocol, TypeAlias, runtime_checkable

from poke_bot.recursive_turn_planner.chance_aware_tree import ChanceAwareSearchConfig


Action: TypeAlias = tuple[int, ...]
TurnKey: TypeAlias = tuple[int, int]

_SCHEMA = "poke_bot.r207_truthful_simulator_replay_arena/v1"
_SHA256_PREFIX = "sha256:"


class R207SimulatorArenaError(RuntimeError):
    """The simulator successor facility cannot make a truthful claim."""


class PlannerDeadlineExceeded(R207SimulatorArenaError):
    """An operation reached an absolute deadline and its result was discarded."""

    def __init__(
        self,
        *,
        operation: str,
        scope: str,
        late_result_discarded: bool,
    ) -> None:
        self.operation = operation
        self.scope = scope
        self.late_result_discarded = late_result_discarded
        detail = "late result discarded" if late_result_discarded else "deadline exhausted"
        super().__init__(f"{scope} deadline exhausted during {operation}; {detail}")


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R207SimulatorArenaError("value is not canonical JSON") from exc
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        raise R207SimulatorArenaError(f"{label} must be a canonical sha256 digest")
    suffix = value[len(_SHA256_PREFIX) :]
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise R207SimulatorArenaError(f"{label} must be a lowercase sha256 digest")
    return value


def _normalise_action(value: Sequence[int] | Action, *, label: str) -> Action:
    if isinstance(value, (str, bytes)):
        raise R207SimulatorArenaError(f"{label} must be a sequence of exact ints")
    try:
        action = tuple(value)
    except TypeError as exc:
        raise R207SimulatorArenaError(f"{label} must be a sequence of exact ints") from exc
    if any(type(option) is not int or option < 0 for option in action):
        raise R207SimulatorArenaError(f"{label} members must be non-negative exact ints")
    if len(set(action)) != len(action):
        raise R207SimulatorArenaError(f"{label} cannot repeat an option")
    return action


def _require_turn_key(value: TurnKey) -> TurnKey:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(part) is not int or part < 0 for part in value)
    ):
        raise R207SimulatorArenaError("turn_key must be a pair of non-negative exact ints")
    return value


def _public_json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy public JSON through its canonical representation.

    The replay classifier receives only this copy.  It cannot receive a native
    battle, a raw opaque snapshot, a pointer, a seed, or a private zone through
    this module's call surface.
    """

    if not isinstance(value, Mapping):
        raise R207SimulatorArenaError("simulator observation is not a public object")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise R207SimulatorArenaError("simulator observation is not canonical JSON") from exc
    if not isinstance(copied, dict):  # defensive; ``value`` was a Mapping.
        raise R207SimulatorArenaError("simulator observation is not an object")
    return copied


def public_observation_sha256(value: Mapping[str, Any]) -> str:
    """Fingerprint exactly the public JSON supplied by a simulator boundary."""

    return _canonical_sha256(_public_json_copy(value))


def _terminal_result_from_public_observation(observation: Mapping[str, Any]) -> int | None:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    result = current.get("result")
    if type(result) is not int or result == -1:
        return None
    return result


class TransitionKind(str, Enum):
    """The only successor classifications accepted by r202/r207."""

    DETERMINISTIC_PUBLIC = "deterministic_public"
    FINITE_PUBLIC_CHANCE = "finite_public_chance"
    INFORMATION_BOUNDARY = "information_boundary"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class OpaqueMidgameHandle:
    """An arena capability, never a serialised engine state.

    ``handle_id`` and the public observation fingerprint are deliberately
    non-secret audit identifiers.  The arena keeps any replay journal and
    snapshot reference privately and rejects forged/unregistered handles.
    """

    arena_id_sha256: str
    handle_id_sha256: str
    public_observation_sha256: str

    def __post_init__(self) -> None:
        _require_digest(self.arena_id_sha256, label="arena_id_sha256")
        _require_digest(self.handle_id_sha256, label="handle_id_sha256")
        _require_digest(self.public_observation_sha256, label="public_observation_sha256")


@dataclass(frozen=True, slots=True)
class ExactChanceOutcome:
    """One opaque child of an exactly enumerated public finite chance node."""

    label: str
    probability: Fraction
    child_handle: OpaqueMidgameHandle
    public_observation_sha256: str
    outcome_certificate_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise R207SimulatorArenaError("finite chance outcome label must be nonempty")
        if not isinstance(self.probability, Fraction) or self.probability <= 0:
            raise R207SimulatorArenaError(
                "finite chance probability must be a positive exact Fraction"
            )
        _require_digest(self.public_observation_sha256, label="chance observation")
        _require_digest(self.outcome_certificate_sha256, label="outcome certificate")


@dataclass(frozen=True, slots=True)
class SuccessorTransition:
    """One classified edge returned by an opaque successor arena."""

    kind: TransitionKind
    transition_certificate_sha256: str
    public_observation_sha256: str | None = None
    child_handle: OpaqueMidgameHandle | None = None
    next_turn_key: TurnKey | None = None
    next_actor_seat: int | None = None
    chance_outcomes: tuple[ExactChanceOutcome, ...] = ()
    terminal_result: int | None = None
    boundary_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransitionKind):
            raise R207SimulatorArenaError("transition kind is invalid")
        _require_digest(self.transition_certificate_sha256, label="transition certificate")
        if self.public_observation_sha256 is not None:
            _require_digest(self.public_observation_sha256, label="transition observation")
        if self.next_turn_key is not None:
            _require_turn_key(self.next_turn_key)
        if self.next_actor_seat is not None and (
            type(self.next_actor_seat) is not int or self.next_actor_seat not in {0, 1}
        ):
            raise R207SimulatorArenaError("next_actor_seat must be 0 or 1")

        if self.kind is TransitionKind.DETERMINISTIC_PUBLIC:
            if (
                self.child_handle is None
                or self.public_observation_sha256 is None
                or self.next_turn_key is None
                or self.next_actor_seat is None
                or self.chance_outcomes
                or self.terminal_result is not None
                or self.boundary_reason is not None
            ):
                raise R207SimulatorArenaError(
                    "deterministic public transition needs exactly one certified child"
                )
        elif self.kind is TransitionKind.FINITE_PUBLIC_CHANCE:
            if (
                self.child_handle is not None
                or self.public_observation_sha256 is not None
                or self.next_turn_key is not None
                or self.next_actor_seat is not None
                or self.terminal_result is not None
                or self.boundary_reason is not None
            ):
                raise R207SimulatorArenaError("finite chance transition has incompatible fields")
            if len(self.chance_outcomes) < 2:
                raise R207SimulatorArenaError(
                    "finite chance requires at least two exact outcomes"
                )
            labels = [outcome.label for outcome in self.chance_outcomes]
            if len(set(labels)) != len(labels):
                raise R207SimulatorArenaError("finite chance outcome labels must be unique")
            if sum((outcome.probability for outcome in self.chance_outcomes), Fraction()) != 1:
                raise R207SimulatorArenaError(
                    "finite chance probabilities must sum exactly to one"
                )
        elif self.kind is TransitionKind.INFORMATION_BOUNDARY:
            if (
                self.child_handle is not None
                or self.public_observation_sha256 is None
                or self.next_turn_key is not None
                or self.next_actor_seat is not None
                or self.chance_outcomes
                or self.terminal_result is not None
                or not isinstance(self.boundary_reason, str)
                or not self.boundary_reason
            ):
                raise R207SimulatorArenaError(
                    "information boundary needs a public observation and exact reason"
                )
        else:  # terminal
            if (
                self.child_handle is not None
                or self.public_observation_sha256 is None
                or self.next_turn_key is not None
                or self.next_actor_seat is not None
                or self.chance_outcomes
                or self.boundary_reason is not None
                or type(self.terminal_result) is not int
                or self.terminal_result < 0
            ):
                raise R207SimulatorArenaError("terminal transition has incompatible fields")


def exact_chance_backup(
    outcomes: Sequence[ExactChanceOutcome],
    values_by_handle_sha256: Mapping[str, Fraction],
) -> Fraction:
    """Back up every finite chance outcome with exact rational arithmetic."""

    values: list[tuple[Fraction, Fraction]] = []
    labels: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, ExactChanceOutcome):
            raise R207SimulatorArenaError("chance outcomes must be typed")
        if outcome.label in labels:
            raise R207SimulatorArenaError("finite chance outcome labels must be unique")
        labels.add(outcome.label)
        try:
            value = values_by_handle_sha256[outcome.child_handle.handle_id_sha256]
        except KeyError as exc:
            raise R207SimulatorArenaError("finite chance child is missing an exact value") from exc
        if not isinstance(value, Fraction):
            raise R207SimulatorArenaError("finite chance child value must be an exact Fraction")
        values.append((outcome.probability, value))
    if len(values) < 2 or sum((probability for probability, _ in values), Fraction()) != 1:
        raise R207SimulatorArenaError("finite chance distribution is incomplete")
    return sum((probability * value for probability, value in values), Fraction())


def controlled_successor_or_boundary(
    transition: SuccessorTransition,
    *,
    controlled_seat: int,
) -> SuccessorTransition:
    """Refuse to search through an opponent's next decision.

    An attested transition can be publicly deterministic yet still end at an
    opponent selection.  Treating that as a normal child would invite an
    opponent-policy call or an implicit private-world assumption.  This helper
    preserves a same-seat deterministic child and turns every other next actor
    into a receipted information boundary.
    """

    if type(controlled_seat) is not int or controlled_seat not in {0, 1}:
        raise R207SimulatorArenaError("controlled_seat must be 0 or 1")
    if transition.kind is not TransitionKind.DETERMINISTIC_PUBLIC:
        return transition
    if transition.next_actor_seat == controlled_seat:
        return transition
    assert transition.public_observation_sha256 is not None
    certificate = _canonical_sha256(
        {
            "schema": _SCHEMA,
            "kind": TransitionKind.INFORMATION_BOUNDARY.value,
            "reason": "opponent_decision_boundary",
            "deterministic_transition_certificate_sha256": (
                transition.transition_certificate_sha256
            ),
            "post_observation_sha256": transition.public_observation_sha256,
            "controlled_seat": controlled_seat,
            "next_actor_seat": transition.next_actor_seat,
        }
    )
    return SuccessorTransition(
        kind=TransitionKind.INFORMATION_BOUNDARY,
        transition_certificate_sha256=certificate,
        public_observation_sha256=transition.public_observation_sha256,
        boundary_reason="opponent_decision_boundary",
    )


@dataclass(frozen=True, slots=True)
class PublicTransitionEvent:
    """The only input a replay classifier receives after a scratch selection."""

    pre_observation: Mapping[str, Any]
    post_observation: Mapping[str, Any]
    action: Action
    pre_observation_sha256: str
    post_observation_sha256: str

    def __post_init__(self) -> None:
        pre = _public_json_copy(self.pre_observation)
        post = _public_json_copy(self.post_observation)
        object.__setattr__(self, "pre_observation", pre)
        object.__setattr__(self, "post_observation", post)
        object.__setattr__(self, "action", _normalise_action(self.action, label="action"))
        if public_observation_sha256(pre) != self.pre_observation_sha256:
            raise R207SimulatorArenaError("pre-observation does not match its fingerprint")
        if public_observation_sha256(post) != self.post_observation_sha256:
            raise R207SimulatorArenaError("post-observation does not match its fingerprint")


@dataclass(frozen=True, slots=True)
class PublicTransitionAttestation:
    """A separately receipted public-only classification for replay v2.

    A replay classifier may attest deterministic public continuation or stop at
    an information boundary.  It may identify a finite chance, but the replay
    arena will convert that to a boundary because v2 cannot enumerate opaque
    outcome children.  It cannot manufacture a terminal result.
    """

    kind: TransitionKind
    certificate_sha256: str
    next_turn_key: TurnKey | None = None
    next_actor_seat: int | None = None
    boundary_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransitionKind):
            raise R207SimulatorArenaError("attestation kind is invalid")
        _require_digest(self.certificate_sha256, label="attestation certificate")
        if self.next_turn_key is not None:
            _require_turn_key(self.next_turn_key)
        if self.next_actor_seat is not None and (
            type(self.next_actor_seat) is not int or self.next_actor_seat not in {0, 1}
        ):
            raise R207SimulatorArenaError("attestation next_actor_seat must be 0 or 1")
        if self.kind is TransitionKind.DETERMINISTIC_PUBLIC:
            if self.next_turn_key is None or self.next_actor_seat is None:
                raise R207SimulatorArenaError(
                    "deterministic attestation needs exact next turn and actor"
                )
            if self.boundary_reason is not None:
                raise R207SimulatorArenaError(
                    "deterministic attestation cannot carry a boundary reason"
                )
        elif self.kind is TransitionKind.INFORMATION_BOUNDARY:
            if not isinstance(self.boundary_reason, str) or not self.boundary_reason:
                raise R207SimulatorArenaError("boundary attestation needs a nonempty reason")
            if self.next_turn_key is not None or self.next_actor_seat is not None:
                raise R207SimulatorArenaError("boundary attestation cannot name a next actor")
        elif self.kind is TransitionKind.FINITE_PUBLIC_CHANCE:
            if self.next_turn_key is not None or self.next_actor_seat is not None:
                raise R207SimulatorArenaError("chance attestation cannot name one next actor")
            if self.boundary_reason is not None:
                raise R207SimulatorArenaError("chance attestation cannot carry a boundary reason")
        else:
            raise R207SimulatorArenaError("replay classifier cannot attest terminal results")


PublicTransitionClassifier: TypeAlias = Callable[
    [PublicTransitionEvent], PublicTransitionAttestation
]


@runtime_checkable
class SuccessorArena(Protocol):
    """Opaque future v3 / current replay arena interface for an MCTS builder."""

    def capture_root(
        self, deadline: "PlannerActionDeadline"
    ) -> tuple[OpaqueMidgameHandle, Mapping[str, Any]]:
        """Capture the current public root as an opaque capability."""

    def observe(
        self, handle: OpaqueMidgameHandle, deadline: "PlannerActionDeadline"
    ) -> Mapping[str, Any]:
        """Return the handle's policy-visible observation only."""

    def expand_action(
        self,
        handle: OpaqueMidgameHandle,
        action: Sequence[int],
        deadline: "PlannerActionDeadline",
    ) -> SuccessorTransition:
        """Run exactly one scratch simulator transition from an opaque node."""


@runtime_checkable
class _RestorableBattle(Protocol):
    def observation(self) -> Mapping[str, Any]:
        ...

    def step(self, action: Sequence[int]) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class _SnapshotRestorer(Protocol):
    def restore_snapshot(self, snapshot: object) -> _RestorableBattle:
        ...


@dataclass(frozen=True, slots=True)
class PlannerActionDeadline:
    """An absolute deadline lease for all work for one atomic action."""

    turn_key: TurnKey
    turn_deadline_ns: int
    action_deadline_ns: int
    _clock_ns: Callable[[], int] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_turn_key(self.turn_key)
        if (
            type(self.turn_deadline_ns) is not int
            or type(self.action_deadline_ns) is not int
            or self.turn_deadline_ns <= 0
            or self.action_deadline_ns <= 0
            or self.action_deadline_ns > self.turn_deadline_ns
        ):
            raise R207SimulatorArenaError("planner deadline lease is malformed")

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.action_deadline_ns - self._clock_ns()) / 1_000_000_000.0

    def _expired_scope(self, now_ns: int) -> str | None:
        if now_ns >= self.turn_deadline_ns:
            return "turn"
        if now_ns >= self.action_deadline_ns:
            return "action"
        return None

    def check(self, operation: str) -> None:
        if not operation:
            raise R207SimulatorArenaError("deadline operation label must be nonempty")
        scope = self._expired_scope(self._clock_ns())
        if scope is not None:
            raise PlannerDeadlineExceeded(
                operation=operation,
                scope=scope,
                late_result_discarded=False,
            )

    def call(
        self,
        operation: str,
        callback: Callable[[], Any],
        *,
        discard_late_result: Callable[[Any], None] | None = None,
    ) -> Any:
        """Call work only while live; discard rather than use a late response.

        Python cannot safely interrupt a blocking native or GPU operation.  The
        caller must therefore already hold the exact direct action.  Once a
        blocking call returns, this method checks the same absolute deadline,
        releases a returned scratch handle when requested, and refuses to use
        the late result.
        """

        self.check(operation)
        value = callback()
        scope = self._expired_scope(self._clock_ns())
        if scope is not None:
            if discard_late_result is not None:
                try:
                    discard_late_result(value)
                except Exception:
                    # The result is already invalid.  Preserve the deadline
                    # reason rather than allowing cleanup to make it usable.
                    pass
            raise PlannerDeadlineExceeded(
                operation=operation,
                scope=scope,
                late_result_discarded=True,
            )
        return value


class AbsolutePlannerDeadlineController:
    """One monotonic turn clock with nested five-second action leases."""

    def __init__(
        self,
        config: ChanceAwareSearchConfig,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(config, ChanceAwareSearchConfig):
            raise R207SimulatorArenaError("r207 deadlines require ChanceAwareSearchConfig")
        if not callable(clock_ns):
            raise R207SimulatorArenaError("clock_ns must be callable")
        self._config = config
        self._clock_ns = clock_ns
        self._active_turn_key: TurnKey | None = None
        self._turn_deadline_ns: int | None = None

    @property
    def config_sha256(self) -> str:
        return self._config.identity_sha256

    def begin_action(self, turn_key: TurnKey) -> PlannerActionDeadline:
        """Open an action lease, resetting the turn only on an exact key change."""

        exact_key = _require_turn_key(turn_key)
        now_ns = self._clock_ns()
        if self._active_turn_key != exact_key:
            self._active_turn_key = exact_key
            self._turn_deadline_ns = now_ns + int(
                float(self._config.max_turn_seconds) * 1_000_000_000
            )
        assert self._turn_deadline_ns is not None
        action_deadline_ns = min(
            self._turn_deadline_ns,
            now_ns + int(float(self._config.max_action_seconds) * 1_000_000_000),
        )
        return PlannerActionDeadline(
            turn_key=exact_key,
            turn_deadline_ns=self._turn_deadline_ns,
            action_deadline_ns=action_deadline_ns,
            _clock_ns=self._clock_ns,
        )


@dataclass(frozen=True, slots=True)
class ReplayJournalEntry:
    """One fully public replay checkpoint; no engine bytes or private zones."""

    ordinal: int
    action: Action
    pre_observation_sha256: str
    post_observation_sha256: str
    deterministic_certificate_sha256: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise R207SimulatorArenaError("replay journal ordinal must be non-negative")
        object.__setattr__(self, "action", _normalise_action(self.action, label="journal action"))
        _require_digest(self.pre_observation_sha256, label="journal pre-observation")
        _require_digest(self.post_observation_sha256, label="journal post-observation")
        _require_digest(
            self.deterministic_certificate_sha256,
            label="journal deterministic certificate",
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "action": list(self.action),
            "pre_observation_sha256": self.pre_observation_sha256,
            "post_observation_sha256": self.post_observation_sha256,
            "deterministic_certificate_sha256": self.deterministic_certificate_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ReplayNode:
    journal: tuple[ReplayJournalEntry, ...]
    public_observation_sha256: str


def _close_quietly(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


class TruthfulReplayArena:
    """Fresh-restore simulator arena for the existing narrow v2 snapshot ABI.

    This is deliberately not called a clone arena.  A node is a sealed root
    plus a verified public journal, and every use replays that journal in a
    fresh scratch battle.  The sealed snapshot itself stays private to the
    instance; the public API exposes only :class:`OpaqueMidgameHandle`.
    """

    def __init__(
        self,
        *,
        engine: _SnapshotRestorer,
        sealed_initial_snapshot: object,
        classify_public_transition: PublicTransitionClassifier,
    ) -> None:
        if not isinstance(engine, _SnapshotRestorer):
            raise R207SimulatorArenaError("engine does not support sealed snapshot restore")
        if not callable(classify_public_transition):
            raise R207SimulatorArenaError("public transition classifier must be callable")
        snapshot_fingerprint = getattr(sealed_initial_snapshot, "fingerprint_sha256", None)
        self._snapshot_fingerprint_sha256 = _require_digest(
            snapshot_fingerprint, label="sealed initial snapshot fingerprint"
        )
        # Intentionally retain only the opaque snapshot object.  Do not read
        # ``serialized_bytes`` (v2's worker-only escape hatch) anywhere here.
        self._snapshot = sealed_initial_snapshot
        self._engine = engine
        self._classifier = classify_public_transition
        self._arena_id_sha256 = _canonical_sha256(
            {
                "schema": _SCHEMA,
                "mode": "sealed_initial_snapshot_fresh_restore_replay",
                "snapshot_fingerprint_sha256": self._snapshot_fingerprint_sha256,
            }
        )
        self._nodes: dict[str, _ReplayNode] = {}
        self._root_handle: OpaqueMidgameHandle | None = None
        self._root_observation_sha256: str | None = None
        self._scratch_restore_count = 0
        self._journal_replay_select_count = 0
        self._scratch_simulator_select_calls = 0
        self._simulator_observations_seen = 0

    @property
    def arena_id_sha256(self) -> str:
        return self._arena_id_sha256

    @property
    def scratch_telemetry(self) -> dict[str, int]:
        """Supplementary raw counters; never substitute for r207 tree counts."""

        return {
            "scratch_restore_count": self._scratch_restore_count,
            "journal_replay_select_count": self._journal_replay_select_count,
            "scratch_simulator_select_calls": self._scratch_simulator_select_calls,
            "simulator_observations_seen": self._simulator_observations_seen,
        }

    def _handle_for_node(self, node: _ReplayNode) -> OpaqueMidgameHandle:
        handle_id = _canonical_sha256(
            {
                "schema": _SCHEMA,
                "arena_id_sha256": self._arena_id_sha256,
                "journal": [entry.as_payload() for entry in node.journal],
                "public_observation_sha256": node.public_observation_sha256,
            }
        )
        self._nodes.setdefault(handle_id, node)
        return OpaqueMidgameHandle(
            arena_id_sha256=self._arena_id_sha256,
            handle_id_sha256=handle_id,
            public_observation_sha256=node.public_observation_sha256,
        )

    def _node_for_handle(self, handle: OpaqueMidgameHandle) -> _ReplayNode:
        if not isinstance(handle, OpaqueMidgameHandle):
            raise R207SimulatorArenaError("simulator node handle is not opaque")
        if handle.arena_id_sha256 != self._arena_id_sha256:
            raise R207SimulatorArenaError("opaque handle belongs to another arena")
        try:
            node = self._nodes[handle.handle_id_sha256]
        except KeyError as exc:
            raise R207SimulatorArenaError("opaque handle is not registered by this arena") from exc
        if node.public_observation_sha256 != handle.public_observation_sha256:
            raise R207SimulatorArenaError("opaque handle public fingerprint is inconsistent")
        return node

    def _restore_and_replay(
        self,
        node: _ReplayNode,
        deadline: PlannerActionDeadline,
    ) -> tuple[_RestorableBattle, dict[str, Any]]:
        battle = deadline.call(
            "simulator_restore",
            lambda: self._engine.restore_snapshot(self._snapshot),
            discard_late_result=_close_quietly,
        )
        self._scratch_restore_count += 1
        try:
            observation = _public_json_copy(
                deadline.call("simulator_observation", battle.observation)
            )
            self._simulator_observations_seen += 1
            observation_sha256 = public_observation_sha256(observation)
            if self._root_observation_sha256 is not None and (
                observation_sha256 != self._root_observation_sha256
            ):
                raise R207SimulatorArenaError(
                    "sealed root restore no longer matches its public fingerprint"
                )
            for entry in node.journal:
                if observation_sha256 != entry.pre_observation_sha256:
                    raise R207SimulatorArenaError(
                        "replay journal pre-observation fingerprint mismatch"
                    )
                observation = _public_json_copy(
                    deadline.call(
                        "simulator_replay_select",
                        lambda entry=entry: battle.step(entry.action),
                    )
                )
                self._journal_replay_select_count += 1
                self._simulator_observations_seen += 1
                observation_sha256 = public_observation_sha256(observation)
                if observation_sha256 != entry.post_observation_sha256:
                    raise R207SimulatorArenaError(
                        "replay journal post-observation fingerprint mismatch"
                    )
            if observation_sha256 != node.public_observation_sha256:
                raise R207SimulatorArenaError("replayed node does not match its public state")
            return battle, observation
        except Exception:
            _close_quietly(battle)
            raise

    def capture_root(
        self, deadline: PlannerActionDeadline
    ) -> tuple[OpaqueMidgameHandle, Mapping[str, Any]]:
        if self._root_handle is not None:
            return self._root_handle, self.observe(self._root_handle, deadline)
        battle = deadline.call(
            "simulator_root_restore",
            lambda: self._engine.restore_snapshot(self._snapshot),
            discard_late_result=_close_quietly,
        )
        self._scratch_restore_count += 1
        try:
            observation = _public_json_copy(
                deadline.call("simulator_root_observation", battle.observation)
            )
            self._simulator_observations_seen += 1
            fingerprint = public_observation_sha256(observation)
            self._root_observation_sha256 = fingerprint
            root = self._handle_for_node(_ReplayNode((), fingerprint))
            self._root_handle = root
            return root, observation
        finally:
            _close_quietly(battle)

    def observe(
        self, handle: OpaqueMidgameHandle, deadline: PlannerActionDeadline
    ) -> Mapping[str, Any]:
        node = self._node_for_handle(handle)
        battle, observation = self._restore_and_replay(node, deadline)
        try:
            return _public_json_copy(observation)
        finally:
            _close_quietly(battle)

    def expand_action(
        self,
        handle: OpaqueMidgameHandle,
        action: Sequence[int],
        deadline: PlannerActionDeadline,
    ) -> SuccessorTransition:
        """Evaluate one scratch edge, refusing any unenumerable successor.

        The live game is never touched.  A deterministic child records a new
        public journal entry.  A chance classification is deliberately turned
        into a boundary because v2 has no opaque child enumerator and the one
        realised scratch outcome must not be mistaken for an expectation.
        """

        node = self._node_for_handle(handle)
        exact_action = _normalise_action(action, label="simulator action")
        battle, before = self._restore_and_replay(node, deadline)
        try:
            before_sha256 = public_observation_sha256(before)
            after = _public_json_copy(
                deadline.call(
                    "simulator_select",
                    lambda: battle.step(exact_action),
                )
            )
            self._scratch_simulator_select_calls += 1
            self._simulator_observations_seen += 1
            after_sha256 = public_observation_sha256(after)
            terminal_result = _terminal_result_from_public_observation(after)
            if terminal_result is not None:
                certificate = _canonical_sha256(
                    {
                        "schema": _SCHEMA,
                        "kind": TransitionKind.TERMINAL.value,
                        "snapshot_fingerprint_sha256": self._snapshot_fingerprint_sha256,
                        "pre_observation_sha256": before_sha256,
                        "action": list(exact_action),
                        "post_observation_sha256": after_sha256,
                        "terminal_result": terminal_result,
                    }
                )
                return SuccessorTransition(
                    kind=TransitionKind.TERMINAL,
                    transition_certificate_sha256=certificate,
                    public_observation_sha256=after_sha256,
                    terminal_result=terminal_result,
                )

            event = PublicTransitionEvent(
                pre_observation=before,
                post_observation=after,
                action=exact_action,
                pre_observation_sha256=before_sha256,
                post_observation_sha256=after_sha256,
            )
            attestation = deadline.call(
                "public_transition_classification",
                lambda: self._classifier(event),
            )
            if not isinstance(attestation, PublicTransitionAttestation):
                raise R207SimulatorArenaError("classifier did not return a typed attestation")

            if attestation.kind is TransitionKind.DETERMINISTIC_PUBLIC:
                entry = ReplayJournalEntry(
                    ordinal=len(node.journal),
                    action=exact_action,
                    pre_observation_sha256=before_sha256,
                    post_observation_sha256=after_sha256,
                    deterministic_certificate_sha256=attestation.certificate_sha256,
                )
                child = self._handle_for_node(
                    _ReplayNode(node.journal + (entry,), after_sha256)
                )
                return SuccessorTransition(
                    kind=TransitionKind.DETERMINISTIC_PUBLIC,
                    transition_certificate_sha256=attestation.certificate_sha256,
                    public_observation_sha256=after_sha256,
                    child_handle=child,
                    next_turn_key=attestation.next_turn_key,
                    next_actor_seat=attestation.next_actor_seat,
                )

            if attestation.kind is TransitionKind.FINITE_PUBLIC_CHANCE:
                # The classifier may identify a chance event, but v2 cannot
                # force/enumerate every outcome.  A sampled scratch successor
                # is not an exact chance child, so it is an explicit boundary.
                certificate = _canonical_sha256(
                    {
                        "schema": _SCHEMA,
                        "kind": TransitionKind.INFORMATION_BOUNDARY.value,
                        "reason": "finite_public_chance_requires_v3_outcome_enumerator",
                        "underlying_attestation_sha256": attestation.certificate_sha256,
                        "post_observation_sha256": after_sha256,
                    }
                )
                return SuccessorTransition(
                    kind=TransitionKind.INFORMATION_BOUNDARY,
                    transition_certificate_sha256=certificate,
                    public_observation_sha256=after_sha256,
                    boundary_reason="finite_public_chance_requires_v3_outcome_enumerator",
                )

            if attestation.kind is TransitionKind.INFORMATION_BOUNDARY:
                return SuccessorTransition(
                    kind=TransitionKind.INFORMATION_BOUNDARY,
                    transition_certificate_sha256=attestation.certificate_sha256,
                    public_observation_sha256=after_sha256,
                    boundary_reason=attestation.boundary_reason,
                )
            raise R207SimulatorArenaError("classifier attempted to manufacture terminal result")
        finally:
            _close_quietly(battle)


@dataclass(slots=True)
class R207TurnSearchLedger:
    """Raw per-turn accounting for a future simulator-backed MCTS builder.

    This ledger never infers missing work.  In particular a fully expanded
    flag can become true only when every declared requested expansion completed
    and was backed up before the deadline.  Normal shadow direct execution is
    distinct from a direct fallback caused by an incomplete/late tree.
    """

    planner_turn_id: str
    seat: int
    turn_key: TurnKey
    selected_action_and_legality_fingerprint: str
    tree_and_config_sha256: str
    actions_dispatched: int = 0
    simulator_transitions_seen: int = 0
    result_or_leaf_evaluations_seen: int = 0
    unique_tree_nodes_seen: int = 1
    decision_nodes_expanded: int = 0
    terminal_exact_results_seen: int = 0
    boundary_leaf_results_seen: int = 0
    neural_leaf_evaluations_seen: int = 0
    finite_chance_outcomes_evaluated: int = 0
    frozen_policy_prior_batches: int = 0
    frozen_policy_prior_evaluations: int = 0
    batched_frozen_outcome_value_leaf_reranking_batches: int = 0
    frozen_outcome_leaf_evaluations: int = 0
    frozen_value_leaf_evaluations: int = 0
    nonterminal_leaves_reranked: int = 0
    terminal_exact_results_not_reranked: bool = True
    cache_hits: int = 0
    deterministic_subtree_reuses: int = 0
    tree_rebuilds: int = 0
    _completed_requested_expansions: int = 0
    _closed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.planner_turn_id, str) or not self.planner_turn_id:
            raise R207SimulatorArenaError("planner_turn_id must be nonempty")
        if type(self.seat) is not int or self.seat not in {0, 1}:
            raise R207SimulatorArenaError("seat must be 0 or 1")
        _require_turn_key(self.turn_key)
        _require_digest(
            self.selected_action_and_legality_fingerprint,
            label="selected action and legality fingerprint",
        )
        _require_digest(self.tree_and_config_sha256, label="tree and config digest")

    def _open(self) -> None:
        if self._closed:
            raise R207SimulatorArenaError("turn search ledger is closed")

    def record_simulator_transition(self, transition: SuccessorTransition) -> None:
        self._open()
        if not isinstance(transition, SuccessorTransition):
            raise R207SimulatorArenaError("simulator transition must be typed")
        self.simulator_transitions_seen += 1
        if transition.kind is TransitionKind.TERMINAL:
            self.terminal_exact_results_seen += 1
            self.result_or_leaf_evaluations_seen += 1
        elif transition.kind is TransitionKind.INFORMATION_BOUNDARY:
            # Boundary is an overlapping classification.  It becomes a
            # result/leaf evaluation only when a later frozen neural leaf
            # batch scores it; do not double-count it here.
            self.boundary_leaf_results_seen += 1
        elif transition.kind is TransitionKind.FINITE_PUBLIC_CHANCE:
            self.finite_chance_outcomes_evaluated += len(transition.chance_outcomes)

    def record_frozen_policy_prior_batch(self, evaluations: int) -> None:
        self._open()
        if type(evaluations) is not int or evaluations <= 0:
            raise R207SimulatorArenaError("prior batch evaluations must be positive exact int")
        self.frozen_policy_prior_batches += 1
        self.frozen_policy_prior_evaluations += evaluations

    def record_frozen_leaf_batch(
        self,
        leaves: int,
        outcome_evals: int,
        value_evals: int,
    ) -> None:
        """Record one truthful batch of frozen nonterminal leaf evaluations.

        The r207 frozen reranker emits both outcome and value outputs for every
        leaf.  ``leaves`` is therefore the number of neural leaf evaluations,
        not the number of batches or model heads.  Boundary leaves may be part
        of this batch, but remain a separate overlapping classification.
        """

        self._open()
        if (
            type(leaves) is not int
            or type(outcome_evals) is not int
            or type(value_evals) is not int
            or leaves <= 0
            or outcome_evals != leaves
            or value_evals != leaves
        ):
            raise R207SimulatorArenaError(
                "frozen leaf batch needs one outcome and value evaluation per exact leaf"
            )
        self.batched_frozen_outcome_value_leaf_reranking_batches += 1
        self.frozen_outcome_leaf_evaluations += outcome_evals
        self.frozen_value_leaf_evaluations += value_evals
        self.neural_leaf_evaluations_seen += leaves
        self.nonterminal_leaves_reranked += leaves
        self.result_or_leaf_evaluations_seen += leaves

    def complete_requested_expansion(self) -> None:
        self._open()
        self._completed_requested_expansions += 1

    def finalize(
        self,
        *,
        requested_expansions: int,
        turn_planner_wall_seconds: float,
        max_single_action_planner_wall_seconds: float,
        deadline_hit: bool,
        tree_incomplete_reason: str | None,
        direct_fallback_used: bool,
        shadow_direct_action: bool,
    ) -> dict[str, object]:
        """Close immutable telemetry without inventing completion or results."""

        self._open()
        if type(requested_expansions) is not int or requested_expansions < 0:
            raise R207SimulatorArenaError("requested expansions must be a non-negative exact int")
        for name, value in (
            ("turn_planner_wall_seconds", turn_planner_wall_seconds),
            ("max_single_action_planner_wall_seconds", max_single_action_planner_wall_seconds),
        ):
            if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
                raise R207SimulatorArenaError(f"{name} must be finite and non-negative")
        if type(deadline_hit) is not bool or type(direct_fallback_used) is not bool:
            raise R207SimulatorArenaError("deadline/fallback flags must be exact booleans")
        if type(shadow_direct_action) is not bool:
            raise R207SimulatorArenaError("shadow direct action flag must be boolean")
        complete = (
            self._completed_requested_expansions == requested_expansions
            and not deadline_hit
            and tree_incomplete_reason is None
        )
        if not complete and not tree_incomplete_reason:
            raise R207SimulatorArenaError("incomplete requested tree needs an exact reason")
        if complete and tree_incomplete_reason is not None:
            raise R207SimulatorArenaError("complete requested tree cannot have an incomplete reason")
        if direct_fallback_used and complete:
            raise R207SimulatorArenaError("complete tree cannot be labelled direct fallback")
        if direct_fallback_used and not (deadline_hit or tree_incomplete_reason):
            raise R207SimulatorArenaError("direct fallback needs an actual incomplete reason")
        if shadow_direct_action and direct_fallback_used:
            raise R207SimulatorArenaError(
                "shadow direct execution and deadline fallback are distinct telemetry states"
            )
        if self.result_or_leaf_evaluations_seen != (
            self.terminal_exact_results_seen + self.neural_leaf_evaluations_seen
        ):
            raise R207SimulatorArenaError(
                "result/leaf count must equal terminal exact results plus neural leaves"
            )
        if (
            self.frozen_outcome_leaf_evaluations != self.neural_leaf_evaluations_seen
            or self.frozen_value_leaf_evaluations != self.neural_leaf_evaluations_seen
            or self.nonterminal_leaves_reranked != self.neural_leaf_evaluations_seen
        ):
            raise R207SimulatorArenaError(
                "frozen outcome/value/nonterminal counts must agree on evaluated leaves"
            )
        self._closed = True
        return {
            "planner_turn_id": self.planner_turn_id,
            "seat": self.seat,
            "turn_key": list(self.turn_key),
            "actions_dispatched": self.actions_dispatched,
            "simulator_transitions_seen": self.simulator_transitions_seen,
            "result_or_leaf_evaluations_seen": self.result_or_leaf_evaluations_seen,
            "unique_tree_nodes_seen": self.unique_tree_nodes_seen,
            "decision_nodes_expanded": self.decision_nodes_expanded,
            "terminal_exact_results_seen": self.terminal_exact_results_seen,
            "boundary_leaf_results_seen": self.boundary_leaf_results_seen,
            "neural_leaf_evaluations_seen": self.neural_leaf_evaluations_seen,
            "finite_chance_outcomes_evaluated": self.finite_chance_outcomes_evaluated,
            "frozen_policy_prior_batches": self.frozen_policy_prior_batches,
            "frozen_policy_prior_evaluations": self.frozen_policy_prior_evaluations,
            "batched_frozen_outcome_value_leaf_reranking_batches": (
                self.batched_frozen_outcome_value_leaf_reranking_batches
            ),
            "frozen_outcome_leaf_evaluations": self.frozen_outcome_leaf_evaluations,
            "frozen_value_leaf_evaluations": self.frozen_value_leaf_evaluations,
            "nonterminal_leaves_reranked": self.nonterminal_leaves_reranked,
            "terminal_exact_results_not_reranked": self.terminal_exact_results_not_reranked,
            "cache_hits": self.cache_hits,
            "deterministic_subtree_reuses": self.deterministic_subtree_reuses,
            "tree_rebuilds": self.tree_rebuilds,
            "turn_planner_wall_seconds": float(turn_planner_wall_seconds),
            "max_single_action_planner_wall_seconds": float(
                max_single_action_planner_wall_seconds
            ),
            "requested_tree_fully_expanded_and_backed_up_within_budget": complete,
            "tree_incomplete_reason": tree_incomplete_reason,
            "deadline_hit": deadline_hit,
            "direct_fallback_used": direct_fallback_used,
            "shadow_direct_action": shadow_direct_action,
            "selected_action_and_legality_fingerprint": self.selected_action_and_legality_fingerprint,
            "tree_and_config_sha256": self.tree_and_config_sha256,
            "completed_requested_expansions": self._completed_requested_expansions,
            "requested_expansions": requested_expansions,
        }


__all__ = [
    "AbsolutePlannerDeadlineController",
    "Action",
    "controlled_successor_or_boundary",
    "ExactChanceOutcome",
    "OpaqueMidgameHandle",
    "PlannerActionDeadline",
    "PlannerDeadlineExceeded",
    "PublicTransitionAttestation",
    "PublicTransitionClassifier",
    "PublicTransitionEvent",
    "R207SimulatorArenaError",
    "R207TurnSearchLedger",
    "ReplayJournalEntry",
    "SuccessorArena",
    "SuccessorTransition",
    "TransitionKind",
    "TruthfulReplayArena",
    "TurnKey",
    "exact_chance_backup",
    "public_observation_sha256",
]
