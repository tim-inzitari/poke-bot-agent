"""Fail-closed staging for the Alakazam matchup-adapter bank.

The adapter bank has two independent gates:

* a *global* boundary receipt proving iteration 15 committed and iteration 16
  is therefore eligible to begin isolated adapter fitting; and
* an offline oracle ticket, derived from an authoritative package/full-deck
  manifest, that may route every causal sequence row while the complete base is
  frozen; and
* a later per-state public-information recognizer for runtime routing.  Oracle
  identity is never an inference input.

This module deliberately does not enable the bank in a serving model.  It
prepares oracle-audited adapter-only fitting rows and a separate public-prefix
audit route while the serialized/runtime feature flag remains disabled.
Promotion and runtime routing are later, separate gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from .matchup_adapters import (
    EXPERT_IDS,
    UNKNOWN_ROUTE,
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
    MatchupAdapterBank,
    route_for_archetype as _v5_route_for_archetype,
)
from .public_matchup_router import visible_opponent_card_ids

if TYPE_CHECKING:  # pragma: no cover - imported only for static checking
    from .dataset import DecisionSample, GameSequence


ACTIVATION_AFTER_COMPLETED_ITERATION = 15
FIRST_ELIGIBLE_ITERATION = 16
ACTIVATION_RECEIPT_SCHEMA = "poke_bot.matchup_adapter_activation_receipt/v1"
ADAPTER_REHEARSAL_AUTHORIZATION_SCHEMA = (
    "poke_bot.matchup_adapter_rehearsal_authorization/v1"
)
SPECIALIST_BOOTSTRAP_AUTHORIZATION_SCHEMA = (
    "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1"
)
CRUSTLE_GUIDE_ALL_EPOCHS_SCHEMA = "poke_bot.crustle_guide_all_epochs/v1"
DEFAULT_SPECIALIST_BOOTSTRAP_EPOCHS = 25
CORPUS_MANIFEST_SCHEMA = "poke_bot.matchup_adapter_corpus/v1"
TRAINING_TICKET_SCHEMA = "poke_bot.matchup_adapter_training_ticket/v1"
PUBLIC_RECOGNIZER_SCHEMA = "poke_bot.public_matchup_recognizer/v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def expected_specialist_bootstrap_epochs(
    training_provenance: Mapping[str, Any],
) -> int:
    """Resolve the owner-authorized bootstrap epoch count for adapter auth.

    Default specialists still use the historical 25-epoch bootstrap. Crustle's
    revision-162 all-guide schedule (`poke_bot.crustle_guide_all_epochs/v1`)
    authorizes the exact closed interval recorded in `guide_active_epochs`.
    """

    guide = dict(training_provenance.get("current_deck_guide") or {})
    owner_schedule = dict(guide.get("owner_epoch_schedule") or {})
    if owner_schedule.get("schema") == CRUSTLE_GUIDE_ALL_EPOCHS_SCHEMA:
        active = list(owner_schedule.get("guide_active_epochs") or ())
        if (
            len(active) == 2
            and int(active[0]) == 1
            and int(active[1]) > 0
        ):
            return int(active[1])
    return DEFAULT_SPECIALIST_BOOTSTRAP_EPOCHS


def training_route_target_ids() -> tuple[str, ...]:
    """Resolve the process-pinned route roster used by offline supervision."""

    if (
        os.environ.get("POKEBOT_MATCHUP_ADAPTER_FORMAT", "").strip()
        != "poke-bot-matchup-adapter-bank-v6"
    ):
        return tuple(EXPERT_IDS)
    from .matchup_adapters_v6 import load_slot_registry

    registry_raw = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH", ""
    ).strip()
    if not registry_raw:
        raise RuntimeError(
            "Router Format 6 adapter training lacks its immutable registry"
        )
    registry = load_slot_registry(Path(registry_raw).expanduser().resolve())
    return tuple(str(value) for value in registry["active_expert_ids"])


def training_route_for_archetype(archetype_id: str | None) -> int:
    """Return the physical training slot under the process-pinned format."""

    if (
        os.environ.get("POKEBOT_MATCHUP_ADAPTER_FORMAT", "").strip()
        != "poke-bot-matchup-adapter-bank-v6"
    ):
        return _v5_route_for_archetype(archetype_id)
    from .matchup_adapters_v6 import load_slot_registry, route_for_archetype

    registry_raw = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH", ""
    ).strip()
    if not registry_raw:
        raise RuntimeError(
            "Router Format 6 adapter training lacks its immutable registry"
        )
    registry = load_slot_registry(Path(registry_raw).expanduser().resolve())
    return route_for_archetype(archetype_id, registry=registry)


# Historical public-router and V5 activation paths retain the legacy mapping.
route_for_archetype = _v5_route_for_archetype


def normalize_matchup_identity(value: Any) -> str:
    """Normalize an identity for collision checks, never fuzzy matching."""

    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def _require_digest(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field_name} must be a canonical sha256 digest")
    return digest


def _canonical_json_digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable JSON receipt without replacing prior evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


# Each tuple is an AND-group; IDs inside one group are alternatives.  Only the
# three evidence-router-v2 rules that passed the held-out precision audit are
# eligible.  Crustle, Cornerstone, Starmie, and Hammer stay permanently UNKNOWN
# until a later recognizer artifact independently passes its gate.  A deck
# classifier's ace card is not sufficient evidence for safe online routing.
PUBLIC_SIGNATURE_GROUPS: Mapping[str, tuple[frozenset[int], ...]] = {
    "marnie-s-grimmsnarl-ex": (frozenset({646, 647, 648}),),
    "garchomp": (frozenset({379, 380, 381, 342, 387, 1173}),),
    "rockets-mewtwo": (frozenset({431}),),
}

UNROUTABLE_EXPERT_IDS = frozenset(EXPERT_IDS) - frozenset(PUBLIC_SIGNATURE_GROUPS)
AUDITED_RUNTIME_EXPERT_IDS = frozenset(PUBLIC_SIGNATURE_GROUPS)
AUDITED_RUNTIME_ROUTES = frozenset(
    route_for_archetype(archetype_id) for archetype_id in AUDITED_RUNTIME_EXPERT_IDS
)
UNACCEPTED_RUNTIME_ROUTES = frozenset(range(len(EXPERT_IDS))) - AUDITED_RUNTIME_ROUTES

# Rejected families still provide negative evidence.  Without this explicit
# conflict tree, a hybrid showing (for example) both Cornerstone and Marnie
# cards could incorrectly route Marnie merely because Cornerstone is disabled.
UNROUTABLE_PUBLIC_CONFLICT_IDS: Mapping[str, frozenset[int]] = {
    "crustle": frozenset({344, 345, 533}),
    "cornerstone-ogerpon": frozenset({117, 386}),
    "starmie": frozenset({1030, 1031}),
    "hammer-pult": frozenset({119, 120, 121}),
}


@dataclass(frozen=True)
class PublicMatchupDecision:
    """One prefix-causal, public-information-only routing decision."""

    route: int = UNKNOWN_ROUTE
    archetype_id: Optional[str] = None
    confidence: float = 0.0
    runner_up_confidence: float = 0.0
    margin: float = 0.0
    consecutive_evidence_states: int = 0
    status: str = "unknown"


@dataclass
class PublicMatchupRecognizer:
    """High-precision stateful decision tree over accumulated public history.

    The recognizer sees only :func:`visible_opponent_card_ids`.  It never reads
    the recorded opponent label, deck, hand, prizes, orchestration metadata, or
    future states.  Recognition requires a complete signature, a unique winner,
    a confidence margin, and repeated evidence-bearing states.  Missing or
    contradictory evidence deactivates immediately; there is no sticky latch.
    """

    confidence_threshold: float = 1.0
    winner_margin: float = 0.50
    consecutive_required: int = 2
    _public_history: set[int] = field(default_factory=set, init=False, repr=False)
    _pending_archetype: Optional[str] = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)
    _last_decision: PublicMatchupDecision = field(
        default_factory=PublicMatchupDecision,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 < float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 <= float(self.winner_margin) <= 1.0:
            raise ValueError("winner_margin must be in [0, 1]")
        if int(self.consecutive_required) < 2:
            raise ValueError("recognition requires at least two evidence states")

    @property
    def last_decision(self) -> PublicMatchupDecision:
        return self._last_decision

    def reset(self) -> None:
        self._public_history.clear()
        self._pending_archetype = None
        self._pending_count = 0
        self._last_decision = PublicMatchupDecision()

    @staticmethod
    def _score(history: set[int], archetype_id: str) -> float:
        groups = PUBLIC_SIGNATURE_GROUPS.get(archetype_id)
        if not groups:
            return 0.0
        matched = sum(bool(history & group) for group in groups)
        return matched / len(groups)

    @staticmethod
    def _current_support(visible: frozenset[int], archetype_id: str) -> bool:
        return any(
            visible & group
            for group in PUBLIC_SIGNATURE_GROUPS.get(archetype_id, ())
        )

    def observe(self, observation: Any) -> PublicMatchupDecision:
        """Consume one state and return its route without inspecting metadata."""

        visible = visible_opponent_card_ids(observation)
        self._public_history.update(visible)
        scored = sorted(
            (
                (self._score(self._public_history, archetype_id), archetype_id)
                for archetype_id in EXPERT_IDS
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, best_id = scored[0]
        runner_score = scored[1][0] if len(scored) > 1 else 0.0
        tied = sum(abs(score - best_score) <= 1e-12 for score, _ in scored) > 1
        margin = best_score - runner_score
        unsupported_conflicts = sorted(
            archetype_id
            for archetype_id, card_ids in UNROUTABLE_PUBLIC_CONFLICT_IDS.items()
            if self._public_history & card_ids
        )
        eligible = bool(
            visible
            and best_score >= float(self.confidence_threshold)
            and margin >= float(self.winner_margin)
            and not tied
            and not unsupported_conflicts
            and self._current_support(visible, best_id)
        )
        if not eligible:
            self._pending_archetype = None
            self._pending_count = 0
            status = (
                "conflict"
                if unsupported_conflicts and best_score > 0.0
                else "ambiguous"
                if tied and best_score > 0.0
                else "unknown"
            )
            self._last_decision = PublicMatchupDecision(
                confidence=float(best_score),
                runner_up_confidence=float(runner_score),
                margin=float(margin),
                status=status,
            )
            return self._last_decision

        if self._pending_archetype == best_id:
            self._pending_count += 1
        else:
            self._pending_archetype = best_id
            self._pending_count = 1
        if self._pending_count < int(self.consecutive_required):
            self._last_decision = PublicMatchupDecision(
                confidence=float(best_score),
                runner_up_confidence=float(runner_score),
                margin=float(margin),
                consecutive_evidence_states=int(self._pending_count),
                status="pending",
            )
            return self._last_decision

        self._last_decision = PublicMatchupDecision(
            route=route_for_archetype(best_id),
            archetype_id=best_id,
            confidence=float(best_score),
            runner_up_confidence=float(runner_score),
            margin=float(margin),
            consecutive_evidence_states=int(self._pending_count),
            status="recognized",
        )
        return self._last_decision

    def state_dict(self) -> dict[str, Any]:
        """Serializable per-game state for packing/checkpoint boundaries."""

        return {
            "schema": PUBLIC_RECOGNIZER_SCHEMA,
            "confidence_threshold": float(self.confidence_threshold),
            "winner_margin": float(self.winner_margin),
            "consecutive_required": int(self.consecutive_required),
            "public_history": sorted(self._public_history),
            "pending_archetype": self._pending_archetype,
            "pending_count": int(self._pending_count),
            "last_decision": {
                "route": int(self._last_decision.route),
                "archetype_id": self._last_decision.archetype_id,
                "confidence": float(self._last_decision.confidence),
                "runner_up_confidence": float(
                    self._last_decision.runner_up_confidence
                ),
                "margin": float(self._last_decision.margin),
                "consecutive_evidence_states": int(
                    self._last_decision.consecutive_evidence_states
                ),
                "status": self._last_decision.status,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if str(state.get("schema") or "") != PUBLIC_RECOGNIZER_SCHEMA:
            raise ValueError("invalid public matchup recognizer state schema")
        if (
            float(state.get("confidence_threshold", -1.0))
            != float(self.confidence_threshold)
            or float(state.get("winner_margin", -1.0))
            != float(self.winner_margin)
            or int(state.get("consecutive_required", -1))
            != int(self.consecutive_required)
        ):
            raise ValueError("public matchup recognizer configuration drift")
        history = {int(value) for value in state.get("public_history") or []}
        if any(value < 0 for value in history):
            raise ValueError("public matchup recognizer history is invalid")
        pending = state.get("pending_archetype")
        if pending is not None and pending not in AUDITED_RUNTIME_EXPERT_IDS:
            raise ValueError(
                "public matchup recognizer pending route is not runtime-authorized"
            )
        pending_count = int(state.get("pending_count", 0))
        if pending_count < 0 or (pending is None and pending_count):
            raise ValueError("public matchup recognizer pending count is invalid")
        decision_payload = dict(state.get("last_decision") or {})
        decision = PublicMatchupDecision(**decision_payload)
        if type(decision.route) is not int:
            raise ValueError("public matchup recognizer saved route is not exact")
        if decision.route != UNKNOWN_ROUTE:
            if decision.archetype_id not in AUDITED_RUNTIME_EXPERT_IDS:
                raise ValueError(
                    "public matchup recognizer saved route is not runtime-authorized"
                )
            if route_for_archetype(decision.archetype_id) != int(decision.route):
                raise ValueError("public matchup recognizer route identity drift")
        self._public_history = history
        self._pending_archetype = pending
        self._pending_count = pending_count
        self._last_decision = decision

    def fork(self) -> "PublicMatchupRecognizer":
        """Copy branch-local state; search branches cannot bleed together."""

        clone = type(self)(
            confidence_threshold=self.confidence_threshold,
            winner_margin=self.winner_margin,
            consecutive_required=self.consecutive_required,
        )
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return clone


@dataclass
class MatchupAdapterGameRouter:
    """Per-game dual gate: fixed boundary eligibility plus public recognizer."""

    first_eligible_iteration: int
    recognizer: PublicMatchupRecognizer = field(default_factory=PublicMatchupRecognizer)

    def __post_init__(self) -> None:
        # Eligibility is fixed when the game begins.  It cannot flip mid-game.
        self._globally_available = (
            int(self.first_eligible_iteration) >= FIRST_ELIGIBLE_ITERATION
        )

    def observe(self, observation: Any) -> PublicMatchupDecision:
        if not self._globally_available:
            return PublicMatchupDecision(status="pre_activation")
        return self.recognizer.observe(observation)

    def reset_for_new_game(self) -> None:
        self.recognizer.reset()

    def fork(self) -> "MatchupAdapterGameRouter":
        clone = type(self)(first_eligible_iteration=self.first_eligible_iteration)
        clone.recognizer = self.recognizer.fork()
        return clone


def runtime_model_route(route: int, *, enabled: bool) -> int:
    """Authorize a public route for serving, or preserve the exact base path.

    Shadow mode always returns :data:`UNKNOWN_ROUTE`, even after the public
    recognizer identifies an audited matchup.  A later serving rollout must
    opt in with ``enabled=True``; doing so for Crustle, Cornerstone Ogerpon,
    Starmie, or Hammer-Pult is a hard error until an independent recognizer
    artifact admits that route.
    """

    if type(route) is not int:
        raise TypeError("runtime matchup route must be an exact integer")
    if type(enabled) is not bool:
        raise TypeError("runtime matchup enable flag must be an exact boolean")
    if route == UNKNOWN_ROUTE:
        return UNKNOWN_ROUTE
    if route < 0 or route >= len(EXPERT_IDS):
        raise ValueError("runtime matchup route is outside the pinned route table")
    if not bool(enabled):
        return UNKNOWN_ROUTE
    if route not in AUDITED_RUNTIME_ROUTES:
        raise RuntimeError(
            f"matchup route {route} ({EXPERT_IDS[route]}) has no accepted "
            "runtime recognizer artifact"
        )
    return route


@dataclass
class MatchupAdapterShadowAudit:
    """Bounded, behavior-inert trace shared by one game and its search forks."""

    max_events: int = 256
    observations: int = 0
    recognized_observations: int = 0
    rejected_observations: int = 0
    per_route: dict[int, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.max_events) < 1:
            raise ValueError("shadow audit max_events must be positive")

    def reset(self) -> None:
        self.observations = 0
        self.recognized_observations = 0
        self.rejected_observations = 0
        self.per_route.clear()
        self.events.clear()

    def record(
        self,
        decision: PublicMatchupDecision,
        *,
        scope: str,
        depth: int,
    ) -> None:
        raw_route = decision.route
        if type(raw_route) is not int:
            raise TypeError("shadow recognizer emitted a non-integer route")
        # This is the central invariant of the shadow rollout: the model route
        # is always UNKNOWN even when the recognizer itself has high-confidence
        # public evidence.
        model_route = runtime_model_route(raw_route, enabled=False)
        if model_route != UNKNOWN_ROUTE:
            raise AssertionError("shadow matchup routing reached the model")
        self.observations += 1
        if raw_route != UNKNOWN_ROUTE:
            self.recognized_observations += 1
            self.per_route[raw_route] = self.per_route.get(raw_route, 0) + 1
            if raw_route not in AUDITED_RUNTIME_ROUTES:
                self.rejected_observations += 1
        if len(self.events) < int(self.max_events):
            self.events.append(
                {
                    "scope": str(scope),
                    "depth": max(0, int(depth)),
                    "status": str(decision.status),
                    "recognized_route": int(raw_route),
                    "model_route": UNKNOWN_ROUTE,
                }
            )

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "poke_bot.matchup_adapter_shadow_audit/v1",
            "mode": "shadow_only",
            "runtime_enabled": False,
            "model_route": UNKNOWN_ROUTE,
            "observations": int(self.observations),
            "recognized_observations": int(self.recognized_observations),
            "rejected_observations": int(self.rejected_observations),
            "per_route": {
                str(route): int(count)
                for route, count in sorted(self.per_route.items())
            },
            "events_truncated": self.observations > len(self.events),
        }
        if include_events:
            payload["events"] = copy.deepcopy(self.events)
        return payload


@dataclass
class ShadowMatchupAdapterRouter:
    """Per-game public recognizer whose only model-facing route is ``-1``.

    Forks copy recognizer history but share the bounded audit sink.  Search
    branches therefore cannot mutate one another's evidence state, while the
    owning game can inspect one consolidated trace after the move.
    """

    game_router: MatchupAdapterGameRouter = field(
        default_factory=lambda: MatchupAdapterGameRouter(
            first_eligible_iteration=FIRST_ELIGIBLE_ITERATION
        )
    )
    audit: MatchupAdapterShadowAudit = field(
        default_factory=MatchupAdapterShadowAudit
    )

    @property
    def model_route(self) -> int:
        return UNKNOWN_ROUTE

    @property
    def candidate_model_route(self) -> int:
        """Return the latest audited public route for a checkpoint-side gate.

        This property does not enable an adapter. The model checkpoint's
        ``matchup_adapters_enabled`` flag remains the final behavior gate, so
        sending this route to a dormant model is an exact no-op.
        """

        return runtime_model_route(
            int(self.game_router.recognizer.last_decision.route),
            enabled=True,
        )

    def observe(
        self,
        observation: Any,
        *,
        scope: str = "game_root",
        depth: int = 0,
    ) -> PublicMatchupDecision:
        decision = self.game_router.observe(observation)
        self.audit.record(decision, scope=scope, depth=depth)
        return decision

    def reset_for_new_game(self) -> None:
        self.game_router.reset_for_new_game()
        self.audit.reset()

    def fork(self) -> "ShadowMatchupAdapterRouter":
        return type(self)(game_router=self.game_router.fork(), audit=self.audit)


@dataclass(frozen=True)
class ActivationReceipt:
    path: Path
    commit_path: Path
    commit_digest: str
    parent_checkpoint: Path
    parent_checkpoint_digest: str
    completed_iteration: int
    first_eligible_iteration: int


def _checkpoint_is_pre_activation_safe(path: Path) -> None:
    """Reject active/non-zero adapters in the frozen iteration-15 parent."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = dict(payload.get("model_config") or {})
    if bool(model_config.get("matchup_adapters_enabled", False)):
        raise RuntimeError("iteration-15 parent already enables matchup adapters")
    extra = dict(payload.get("extra") or {})
    if extra.get("matchup_adapters_runtime_enabled", False) is not False:
        raise RuntimeError(
            "iteration-15 parent ambiguously marks matchup adapters runtime-active"
        )
    dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
    if dormant.get("runtime_enabled", False) is not False:
        raise RuntimeError(
            "iteration-15 parent dormant-adapter record is runtime-active"
        )
    state = dict(payload.get("model_state_dict") or {})
    adapter_tensors = {
        key: value
        for key, value in state.items()
        if "matchup_adapter_bank." in key
    }
    # A prior dormant integration may have emitted the exact zero-output bank.
    # Down projections can be non-zero, but every up projection must still be
    # exactly zero for the bank to be a guaranteed no-op.
    unsafe = [
        key
        for key, value in adapter_tensors.items()
        if (".up.weight" in key or ".up.bias" in key)
        and int(value.detach().count_nonzero().item()) != 0
    ]
    if unsafe:
        raise RuntimeError(
            "iteration-15 parent contains non-dormant adapter outputs: "
            f"{unsafe[:4]}"
        )


def _iteration16_has_started(run_dir: Path) -> bool:
    """Detect every durable start marker used before/during iteration 16.

    The collection shard may still be empty immediately after ``_kick_collect``;
    ``iteration_runtime.json`` is therefore part of the assertion, alongside
    final and temporary iteration artifacts.
    """

    stem = "iter_00016"
    fixed = (
        run_dir / "collection_receipts" / f"{stem}.json",
        run_dir / "shards" / f"{stem}.jsonl",
        run_dir / "checkpoints" / f"{stem}.pt",
        run_dir / "commits" / f"{stem}.json",
        run_dir / "eval" / f"{stem}.json",
        run_dir / "metrics" / f"{stem}.json",
    )
    if any(path.exists() for path in fixed):
        return True
    for parent in (run_dir / "shards", run_dir / "checkpoints"):
        if parent.is_dir() and any(parent.glob(f"*{stem}*")):
            return True
    runtime_path = run_dir / "iteration_runtime.json"
    if runtime_path.is_file():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A torn/ambiguous runtime pointer cannot prove a clean boundary.
            return True
        if int(runtime.get("iteration", -1)) >= FIRST_ELIGIBLE_ITERATION:
            return True
    latest_metrics = run_dir / "metrics" / "latest.json"
    if latest_metrics.is_file():
        try:
            latest = json.loads(latest_metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if int(latest.get("iteration", -1)) >= FIRST_ELIGIBLE_ITERATION:
            return True
    return False


def build_activation_receipt(
    *,
    run_dir: Path,
    output_path: Path,
) -> ActivationReceipt:
    """Create an immutable proof tied to the committed iteration-15 learner.

    The function never edits the run ledger, checkpoint, service, or active
    process.  A missing/malformed commit, mutable pointer disagreement at the
    exact boundary, checkpoint digest mismatch, or active parent fails closed.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    commit_path = run_dir / "commits" / "iter_00015.json"
    if not commit_path.is_file():
        raise RuntimeError("iteration 15 has not committed")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if (
        int(commit.get("last_completed_iteration", -1))
        != ACTIVATION_AFTER_COMPLETED_ITERATION
        or int(commit.get("next_iteration", -1)) != FIRST_ELIGIBLE_ITERATION
    ):
        raise RuntimeError("iteration-15 commit is not the exact 15 -> 16 boundary")
    loop_path = run_dir / "loop_state.json"
    if not loop_path.is_file():
        raise RuntimeError("pure-RL loop ledger is missing at adapter boundary")
    loop_state = json.loads(loop_path.read_text(encoding="utf-8"))
    if (
        int(loop_state.get("last_completed_iteration", -1))
        != ACTIVATION_AFTER_COMPLETED_ITERATION
        or int(loop_state.get("next_iteration", -1)) != FIRST_ELIGIBLE_ITERATION
        or loop_state != commit
    ):
        raise RuntimeError(
            "adapter activation must be staged at the exact committed 15 -> 16 "
            "ledger boundary"
        )
    if _iteration16_has_started(run_dir):
        raise RuntimeError("iteration 16 already started; boundary staging is closed")
    learner = dict(commit.get("learner") or {})
    parent = Path(str(learner.get("path") or "")).expanduser().resolve()
    parent_digest = _require_digest(
        learner.get("digest"), field_name="commit learner digest"
    )
    if not parent.is_file() or _file_digest(parent) != parent_digest:
        raise RuntimeError("iteration-15 learner checkpoint identity mismatch")
    _checkpoint_is_pre_activation_safe(parent)

    # Re-read the mutable ledger immediately before publishing the receipt.
    # This closes the practical TOCTOU window for the non-orchestrating CLI;
    # the in-loop hook is already serialized before ``_kick_collect(16)``.
    try:
        final_loop_state = json.loads(loop_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("pure-RL loop ledger changed during boundary staging") from exc
    if final_loop_state != commit or _iteration16_has_started(run_dir):
        raise RuntimeError("iteration 16 started during boundary staging")

    payload = {
        "schema": ACTIVATION_RECEIPT_SCHEMA,
        "completed_iteration": ACTIVATION_AFTER_COMPLETED_ITERATION,
        "first_eligible_iteration": FIRST_ELIGIBLE_ITERATION,
        "commit_path": str(commit_path),
        "commit_digest": _file_digest(commit_path),
        "loop_state_path": str(loop_path),
        "loop_state_digest": _file_digest(loop_path),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_digest": parent_digest,
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "parent_untouched": True,
    }
    _atomic_json_exclusive(Path(output_path), payload)
    return validate_activation_receipt(output_path, parent_checkpoint=parent)


def validate_activation_receipt(
    path: Path | str,
    *,
    parent_checkpoint: Path | str,
) -> ActivationReceipt:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema") or "") != ACTIVATION_RECEIPT_SCHEMA:
        raise ValueError("invalid matchup-adapter activation receipt schema")
    if (
        int(payload.get("completed_iteration", -1))
        != ACTIVATION_AFTER_COMPLETED_ITERATION
        or int(payload.get("first_eligible_iteration", -1))
        != FIRST_ELIGIBLE_ITERATION
        or payload.get("runtime_enabled") is not False
        or payload.get("optimizer_scope") != "matchup_adapter_bank_only"
        or payload.get("parent_untouched") is not True
    ):
        raise ValueError("matchup-adapter activation receipt contract mismatch")
    commit_path = Path(str(payload.get("commit_path") or "")).expanduser().resolve()
    if not commit_path.is_file() or _file_digest(commit_path) != _require_digest(
        payload.get("commit_digest"), field_name="receipt commit digest"
    ):
        raise ValueError("matchup-adapter boundary commit identity changed")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if (
        int(commit.get("last_completed_iteration", -1))
        != ACTIVATION_AFTER_COMPLETED_ITERATION
        or int(commit.get("next_iteration", -1)) != FIRST_ELIGIBLE_ITERATION
    ):
        raise ValueError("matchup-adapter boundary commit is no longer valid")
    expected_parent = Path(parent_checkpoint).expanduser().resolve()
    saved_parent = Path(
        str(payload.get("parent_checkpoint") or "")
    ).expanduser().resolve()
    if expected_parent != saved_parent:
        raise ValueError("activation receipt is pinned to a different parent")
    parent_digest = _require_digest(
        payload.get("parent_checkpoint_digest"),
        field_name="receipt parent digest",
    )
    safety_parent = saved_parent
    if not saved_parent.is_file():
        # Artifact retention may retire the bankless parent after an immutable
        # zero-dormant copy was materialized at the verified boundary. Accept
        # only that deterministic sibling and only when its frozen provenance
        # cryptographically binds both the missing parent and this receipt.
        materialized = saved_parent.with_name(
            saved_parent.stem + ".zero-dormant.pt"
        )
        if not materialized.is_file():
            raise ValueError("activation receipt parent checkpoint identity changed")
        from .dormant_adapter_compat import validate_zero_dormant_checkpoint
        from . import checkpoint as checkpoint_mod

        validate_zero_dormant_checkpoint(materialized)
        materialized_payload = checkpoint_mod.load_checkpoint(
            materialized, map_location="cpu"
        )
        dormant = dict(
            (materialized_payload.get("extra") or {}).get(
                "dormant_matchup_adapter_bank"
            )
            or {}
        )
        if (
            Path(str(dormant.get("parent_checkpoint") or ""))
            .expanduser()
            .resolve()
            != saved_parent
            or str(dormant.get("parent_checkpoint_digest") or "").lower()
            != parent_digest
            or Path(str(dormant.get("activation_receipt") or ""))
            .expanduser()
            .resolve()
            != path
            or str(dormant.get("activation_receipt_digest") or "").lower()
            != _file_digest(path)
            or dormant.get("materialization")
            not in {
                "immutable_bankless_parent_copy",
                "legacy_bankless_dynamic_zero_init",
            }
        ):
            raise ValueError(
                "materialized zero-dormant parent does not prove the receipt identity"
            )
        safety_parent = materialized
    elif _file_digest(saved_parent) != parent_digest:
        raise ValueError("activation receipt parent checkpoint identity changed")
    learner = dict(commit.get("learner") or {})
    if (
        Path(str(learner.get("path") or "")).expanduser().resolve() != saved_parent
        or str(learner.get("digest") or "").lower() != parent_digest
    ):
        raise ValueError("activation receipt parent is not the committed learner")
    _checkpoint_is_pre_activation_safe(safety_parent)
    return ActivationReceipt(
        path=path,
        commit_path=commit_path,
        commit_digest=str(payload["commit_digest"]),
        parent_checkpoint=saved_parent,
        parent_checkpoint_digest=parent_digest,
        completed_iteration=ACTIVATION_AFTER_COMPLETED_ITERATION,
        first_eligible_iteration=FIRST_ELIGIBLE_ITERATION,
    )


def build_adapter_rehearsal_authorization(
    *,
    run_dir: Path,
    completed_iteration: int,
    output_path: Path,
) -> ActivationReceipt:
    """Authorize adapter-only fitting at one exact committed RL boundary."""

    run_dir = Path(run_dir).expanduser().resolve()
    completed = int(completed_iteration)
    if completed < 0:
        raise ValueError("adapter fitting requires a completed RL iteration")
    commit_path = run_dir / "commits" / f"iter_{completed:05d}.json"
    loop_path = run_dir / "loop_state.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    loop_state = json.loads(loop_path.read_text(encoding="utf-8"))
    if (
        commit != loop_state
        or int(commit.get("last_completed_iteration", -1)) != completed
        or int(commit.get("next_iteration", -1)) != completed + 1
    ):
        raise RuntimeError("adapter rehearsal requires an exact committed boundary")
    if _iteration_artifacts_exist(run_dir, completed + 1):
        raise RuntimeError("the next RL iteration already has durable artifacts")
    learner = dict(commit.get("learner") or commit.get("champion") or {})
    parent = Path(str(learner.get("path") or "")).expanduser().resolve()
    parent_digest = _require_digest(
        learner.get("digest"), field_name="boundary learner digest"
    )
    if not parent.is_file() or _file_digest(parent) != parent_digest:
        raise RuntimeError("boundary learner checkpoint identity mismatch")
    payload = {
        "schema": ADAPTER_REHEARSAL_AUTHORIZATION_SCHEMA,
        "completed_iteration": completed,
        "first_eligible_iteration": completed + 1,
        "commit_path": str(commit_path),
        "commit_digest": _file_digest(commit_path),
        "loop_state_path": str(loop_path),
        "loop_state_digest": _file_digest(loop_path),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_digest": parent_digest,
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "parent_untouched": True,
        "purpose": "specialist-causal-router-aligned-adapter-fitting",
    }
    # Close the same practical TOCTOU window as the initial activation path.
    if (
        json.loads(loop_path.read_text(encoding="utf-8")) != commit
        or _iteration_artifacts_exist(run_dir, completed + 1)
    ):
        raise RuntimeError("the next RL iteration started during authorization")
    _atomic_json_exclusive(Path(output_path), payload)
    return validate_adapter_training_authorization(
        output_path, parent_checkpoint=parent
    )


def _iteration_artifacts_exist(run_dir: Path, iteration: int) -> bool:
    stem = f"iter_{int(iteration):05d}"
    paths = (
        run_dir / "collection_receipts" / f"{stem}.json",
        run_dir / "shards" / f"{stem}.jsonl",
        run_dir / "checkpoints" / f"{stem}.pt",
        run_dir / "commits" / f"{stem}.json",
        run_dir / "eval" / f"{stem}.json",
        run_dir / "metrics" / f"{stem}.json",
    )
    if any(path.exists() for path in paths):
        return True
    for parent in (run_dir / "shards", run_dir / "checkpoints"):
        if parent.is_dir() and any(parent.glob(f"*{stem}*")):
            return True
    for pointer in (
        run_dir / "iteration_runtime.json",
        run_dir / "metrics" / "latest.json",
    ):
        if not pointer.is_file():
            continue
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if int(payload.get("iteration", -1)) >= int(iteration):
            return True
    return False


def validate_adapter_training_authorization(
    path: Path | str,
    *,
    parent_checkpoint: Path | str,
    permit_post_boundary_use: bool = False,
) -> ActivationReceipt:
    """Validate either the initial activation or a later boundary receipt.

    ``permit_post_boundary_use`` is only for the adapter-only optimizer that
    necessarily runs after the authorized iteration has created collection
    artifacts.  In that mode the immutable commit and the receipt's recorded
    clean-boundary snapshot remain mandatory, while the *live* loop pointer is
    allowed to have advanced.  The default remains the stricter pre-start
    check used when creating or staging an authorization.
    """

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") == ACTIVATION_RECEIPT_SCHEMA:
        return validate_activation_receipt(
            resolved, parent_checkpoint=parent_checkpoint
        )
    if payload.get("schema") == SPECIALIST_BOOTSTRAP_AUTHORIZATION_SCHEMA:
        parent = Path(parent_checkpoint).expanduser().resolve()
        authorized_parent = Path(
            str(payload.get("parent_checkpoint") or "")
        ).expanduser().resolve()
        parent_digest = _require_digest(
            payload.get("parent_checkpoint_digest"),
            field_name="authorization parent digest",
        )
        protected_manifest = Path(
            str(payload.get("protected_manifest") or "")
        ).expanduser().resolve()
        protected_manifest_digest = _require_digest(
            payload.get("protected_manifest_digest"),
            field_name="protected manifest digest",
        )
        manifest = json.loads(protected_manifest.read_text(encoding="utf-8"))
        provenance = dict(manifest.get("provenance") or {})
        evidence = dict(manifest.get("evidence") or {})
        training_manifest = manifest
        if provenance.get("kind") == "matchup_adapter_v6_runtime_derivative":
            source_family = Path(
                str(provenance.get("source_family") or "")
            ).expanduser().resolve()
            source_manifest = source_family / "manifest.json"
            source_checkpoint = source_family / "model.pt"
            source_manifest_digest = _require_digest(
                provenance.get("source_family_manifest_sha256"),
                field_name="source family manifest digest",
            )
            source_checkpoint_digest = _require_digest(
                provenance.get("source_checkpoint_digest"),
                field_name="source checkpoint digest",
            )
            if (
                evidence.get("training_evidence_inherited_from_source")
                is not True
                or provenance.get("source_family_immutable") is not True
                or not source_manifest.is_file()
                or _file_digest(source_manifest) != source_manifest_digest
                or not source_checkpoint.is_file()
                or _file_digest(source_checkpoint)
                != source_checkpoint_digest
            ):
                raise ValueError(
                    "specialist bootstrap inherited training evidence is invalid"
                )
            training_manifest = json.loads(
                source_manifest.read_text(encoding="utf-8")
            )
            if (
                str(training_manifest.get("checkpoint_digest") or "").lower()
                != source_checkpoint_digest
                or Path(
                    str(training_manifest.get("model_path") or "")
                ).expanduser().resolve()
                != source_checkpoint
            ):
                raise ValueError(
                    "specialist bootstrap inherited training evidence is invalid"
                )
        training_provenance = dict(
            training_manifest.get("provenance") or {}
        )
        training_evidence = dict(training_manifest.get("evidence") or {})
        specialist_id = normalize_matchup_identity(payload.get("specialist_id"))
        required_target_coverage = tuple(
            str(value)
            for value in payload.get("required_target_coverage") or ()
        )
        trained_target_coverage = tuple(
            str(value)
            for value in training_provenance.get(
                "trained_target_coverage"
            ) or ()
        )
        sparse_bootstrap_coverage_valid = bool(
            required_target_coverage
            and required_target_coverage == trained_target_coverage
            and "temporal_action_rows" in required_target_coverage
            and len(required_target_coverage)
            == len(set(required_target_coverage))
        )
        expected_epochs = expected_specialist_bootstrap_epochs(
            training_provenance
        )
        if (
            not specialist_id
            or payload.get("runtime_enabled") is not False
            or payload.get("parent_untouched") is not True
            or payload.get("optimizer_scope") != "matchup_adapter_bank_only"
            or int(payload.get("first_eligible_iteration", -1)) != 0
            or int(payload.get("completed_iteration", -2)) != -1
            or authorized_parent != parent
            or not parent.is_file()
            or _file_digest(parent) != parent_digest
            or not protected_manifest.is_file()
            or _file_digest(protected_manifest) != protected_manifest_digest
            or str(manifest.get("checkpoint_digest") or "").lower()
            != parent_digest
            or Path(str(manifest.get("model_path") or "")).expanduser().resolve()
            != parent
            or normalize_matchup_identity(
                training_provenance.get("acting_seat_archetype")
            )
            != specialist_id
            or int(training_evidence.get("epochs_completed", -1))
            != expected_epochs
            or int(training_provenance.get("epochs_max", -1))
            != expected_epochs
            or (
                training_provenance.get("all_auxiliary_heads_trained")
                is not True
                and not sparse_bootstrap_coverage_valid
            )
        ):
            raise ValueError(
                "specialist bootstrap adapter authorization contract is invalid"
            )
        return ActivationReceipt(
            path=resolved,
            commit_path=protected_manifest,
            commit_digest=protected_manifest_digest,
            parent_checkpoint=parent,
            parent_checkpoint_digest=parent_digest,
            completed_iteration=-1,
            first_eligible_iteration=0,
        )
    if payload.get("schema") != ADAPTER_REHEARSAL_AUTHORIZATION_SCHEMA:
        raise ValueError("invalid matchup-adapter training authorization schema")
    parent = Path(parent_checkpoint).expanduser().resolve()
    completed = int(payload.get("completed_iteration", -1))
    first_eligible = int(payload.get("first_eligible_iteration", -1))
    commit_path = Path(str(payload.get("commit_path") or "")).expanduser().resolve()
    if (
        completed < 0
        or first_eligible != completed + 1
        or payload.get("runtime_enabled") is not False
        or payload.get("parent_untouched") is not True
        or payload.get("optimizer_scope") != "matchup_adapter_bank_only"
        or commit_path.name != f"iter_{completed:05d}.json"
        or not commit_path.is_file()
        or _file_digest(commit_path)
        != _require_digest(payload.get("commit_digest"), field_name="commit digest")
    ):
        raise ValueError("adapter rehearsal authorization contract is invalid")
    authorized_parent = Path(
        str(payload.get("parent_checkpoint") or "")
    ).expanduser().resolve()
    parent_digest = _require_digest(
        payload.get("parent_checkpoint_digest"),
        field_name="authorization parent digest",
    )
    if (
        authorized_parent != parent
        or not parent.is_file()
        or _file_digest(parent) != parent_digest
    ):
        raise ValueError("adapter rehearsal parent checkpoint identity mismatch")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    loop_path = Path(
        str(payload.get("loop_state_path") or "")
    ).expanduser().resolve()
    loop_digest = _require_digest(
        payload.get("loop_state_digest"), field_name="loop-state digest"
    )
    if permit_post_boundary_use:
        # At authorization time loop_state and the immutable commit were
        # byte-identical.  Preserve that issuance proof without requiring the
        # live loop pointer to remain frozen after collection has started.
        if loop_digest != _file_digest(commit_path):
            raise ValueError("adapter rehearsal clean-boundary proof is invalid")
    elif (
        not loop_path.is_file()
        or _file_digest(loop_path) != loop_digest
        or json.loads(loop_path.read_text(encoding="utf-8")) != commit
        or _iteration_artifacts_exist(commit_path.parent.parent, first_eligible)
    ):
        raise ValueError("adapter rehearsal boundary is no longer clean")
    learner = dict(commit.get("learner") or commit.get("champion") or {})
    if (
        int(commit.get("last_completed_iteration", -1)) != completed
        or int(commit.get("next_iteration", -1)) != first_eligible
        or Path(str(learner.get("path") or "")).expanduser().resolve() != parent
        or str(learner.get("digest") or "").lower() != parent_digest
    ):
        raise ValueError("adapter rehearsal authorization no longer matches its commit")
    return ActivationReceipt(
        path=resolved,
        commit_path=commit_path,
        commit_digest=_file_digest(commit_path),
        parent_checkpoint=parent,
        parent_checkpoint_digest=parent_digest,
        completed_iteration=completed,
        first_eligible_iteration=first_eligible,
    )


def materialize_zero_dormant_adapter_checkpoint(
    *,
    parent_checkpoint: Path | str,
    activation_receipt: Path | str,
    output_path: Path | str,
) -> Path:
    """Add a frozen, exactly zero-output bank to one receipt-pinned parent.

    This is an architecture-compatibility migration, not adapter fitting and
    not runtime activation.  Every pre-existing model tensor and every other
    top-level checkpoint field is preserved.  Only the model-config spelling,
    the complete zero-output bank, and explicit dormant provenance are added.
    The output is published immutably and the parent is never modified.
    """

    import torch

    from . import checkpoint as checkpoint_mod

    parent_path = Path(parent_checkpoint).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == parent_path:
        raise ValueError("zero-bank output must not replace its parent")
    proof = validate_activation_receipt(
        activation_receipt,
        parent_checkpoint=parent_path,
    )
    parent = checkpoint_mod.load_checkpoint(parent_path, map_location="cpu")
    parent_state = dict(parent.get("model_state_dict") or {})
    existing_adapter_keys = sorted(
        name
        for name in parent_state
        if name.startswith("matchup_adapter_bank.")
    )
    if existing_adapter_keys:
        raise RuntimeError(
            "zero-bank materialization requires a bankless parent; found "
            f"{existing_adapter_keys[:4]}"
        )
    model_config = dict(parent.get("model_config") or {})
    if bool(model_config.get("matchup_adapters_enabled", False)):
        raise RuntimeError("parent checkpoint already enables matchup adapters")
    extra = dict(parent.get("extra") or {})
    if extra.get("matchup_adapters_runtime_enabled", False) is not False:
        raise RuntimeError("parent checkpoint ambiguously enables matchup adapters")
    if extra.get("matchup_adapter_config") is not None or extra.get(
        "dormant_matchup_adapter_bank"
    ) is not None:
        raise RuntimeError("bankless parent has conflicting adapter metadata")

    # Deterministic down-projection initialization makes the materialized file
    # reproducible without consuming or changing the caller's Torch RNG state.
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        bank = MatchupAdapterBank(enabled=False)
    finally:
        torch.random.set_rng_state(rng_state)
    bank.requires_grad_(False)
    bank_state = {
        f"matchup_adapter_bank.{name}": value.detach().cpu().clone()
        for name, value in bank.state_dict().items()
    }
    unsafe_outputs = [
        name
        for name, value in bank_state.items()
        if (name.endswith(".up.weight") or name.endswith(".up.bias"))
        and int(value.count_nonzero().item()) != 0
    ]
    if unsafe_outputs:
        raise RuntimeError(
            f"fresh dormant bank is not zero-output: {unsafe_outputs[:4]}"
        )

    migrated = copy.deepcopy(parent)
    migrated_state = dict(migrated.get("model_state_dict") or {})
    migrated_state.update(bank_state)
    migrated["model_state_dict"] = migrated_state
    migrated_config = dict(migrated.get("model_config") or {})
    migrated_config["matchup_adapters_enabled"] = False
    migrated["model_config"] = migrated_config
    migrated_extra = dict(migrated.get("extra") or {})
    migrated_extra["matchup_adapter_config"] = bank.config_dict()
    migrated_extra["matchup_adapters_runtime_enabled"] = False
    migrated_extra["matchup_adapter_training_enabled"] = False
    migrated_extra["matchup_adapter_optimizer_included"] = False
    migrated_extra["dormant_matchup_adapter_bank"] = {
        "schema": ZERO_DORMANT_CHECKPOINT_SCHEMA,
        "materialization": "immutable_bankless_parent_copy",
        "runtime_enabled": False,
        "training_enabled": False,
        "optimizer_imported": False,
        "optimizer_present": "optimizer_state_dict" in parent,
        "optimizer_included": False,
        "frozen": True,
        "zero_output": True,
        "parameter_count": sum(value.numel() for value in bank_state.values()),
        "activation_receipt": str(proof.path),
        "activation_receipt_digest": _file_digest(proof.path),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_digest": proof.parent_checkpoint_digest,
        "adapter_config": bank.config_dict(),
    }
    migrated["extra"] = migrated_extra

    # Check the preservation claim before any artifact is published.
    for name, value in parent_state.items():
        candidate = migrated_state.get(name)
        if not isinstance(value, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            raise RuntimeError(f"model state is not tensor-valued: {name}")
        if not torch.equal(value, candidate):
            raise RuntimeError(f"zero-bank migration changed base tensor: {name}")

    checkpoint_mod.immutable_torch_save(migrated, output)
    saved = checkpoint_mod.load_checkpoint(output, map_location="cpu")
    saved_state = dict(saved.get("model_state_dict") or {})
    for name, value in parent_state.items():
        if name not in saved_state or not torch.equal(value, saved_state[name]):
            raise RuntimeError(
                f"published zero-bank checkpoint changed base tensor: {name}"
            )
    saved_extra = dict(saved.get("extra") or {})
    if (
        bool(dict(saved.get("model_config") or {}).get("matchup_adapters_enabled"))
        or saved_extra.get("matchup_adapters_runtime_enabled") is not False
        or saved_extra.get("matchup_adapter_training_enabled") is not False
        or saved_extra.get("matchup_adapter_optimizer_included") is not False
        or dict(saved_extra.get("dormant_matchup_adapter_bank") or {}).get(
            "training_enabled"
        )
        is not False
    ):
        raise RuntimeError("published dormant checkpoint is not runtime/training-off")
    return output


def merge_dormant_adapter_checkpoint(
    *,
    parent_checkpoint: Path | str,
    adapter_checkpoint: Path | str,
    activation_receipt: Path | str,
    output_path: Path | str,
    permit_post_boundary_use: bool = False,
    import_optimizer_state: bool = False,
    accumulate_parent_fit: bool = False,
) -> Path:
    """Merge only trained adapter tensors into the exact parent payload.

    The parent optimizer/scaler/scheduler/RNG/counters and every non-adapter
    tensor are preserved byte-for-byte in memory.  By default the standalone
    adapter optimizer is discarded for compatibility with the initial
    materialization path.  A receipt-gated rehearsal may explicitly import
    that isolated optimizer and accumulate prior fit totals.  The merged
    checkpoint remains runtime-off; it is a dormant continuation candidate,
    not a promoted policy.
    """

    import torch

    from . import checkpoint as checkpoint_mod

    parent_path = Path(parent_checkpoint).expanduser().resolve()
    adapter_path = Path(adapter_checkpoint).expanduser().resolve()
    proof = validate_adapter_training_authorization(
        activation_receipt,
        parent_checkpoint=parent_path,
        permit_post_boundary_use=bool(permit_post_boundary_use),
    )
    parent = checkpoint_mod.load_checkpoint(parent_path, map_location="cpu")
    trained = checkpoint_mod.load_checkpoint(adapter_path, map_location="cpu")
    trained_extra = dict(trained.get("extra") or {})
    receipt_digest = _file_digest(proof.path)
    saved_receipt_path = Path(
        str(trained_extra.get("matchup_adapter_activation_receipt") or "")
    ).expanduser().resolve()
    saved_parent_path = Path(
        str(trained_extra.get("matchup_adapter_parent_checkpoint") or "")
    ).expanduser().resolve()
    if (
        str(trained_extra.get("matchup_adapter_parent_checkpoint_digest") or "")
        != proof.parent_checkpoint_digest
        or saved_parent_path != parent_path
        or saved_receipt_path != proof.path
        or str(
            trained_extra.get("matchup_adapter_activation_receipt_digest") or ""
        )
        != receipt_digest
        or bool(
            dict(trained.get("model_config") or {}).get(
                "matchup_adapters_enabled", False
            )
        )
        or trained_extra.get("matchup_adapters_runtime_enabled") is not False
    ):
        raise RuntimeError("adapter checkpoint is not a dormant child of the receipt")

    parent_state = dict(parent.get("model_state_dict") or {})
    trained_state = dict(trained.get("model_state_dict") or {})
    parent_base = {
        name: value
        for name, value in parent_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    trained_base = {
        name: value
        for name, value in trained_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    if parent_base.keys() != trained_base.keys():
        raise RuntimeError("adapter checkpoint base key set differs from parent")
    changed = [
        name
        for name in parent_base
        if not torch.equal(parent_base[name], trained_base[name])
    ]
    if changed:
        raise RuntimeError(
            "adapter checkpoint changed frozen parent tensors: " f"{changed[:5]}"
        )
    adapter_state = {
        name: value
        for name, value in trained_state.items()
        if name.startswith("matchup_adapter_bank.")
    }
    expected_config = trained_extra.get("matchup_adapter_config")
    if expected_config != MatchupAdapterBank.config_dict():
        raise RuntimeError("trained checkpoint lacks adapter state/routing contract")
    expected_bank = MatchupAdapterBank(enabled=False).state_dict()
    expected_state = {
        f"matchup_adapter_bank.{name}": value for name, value in expected_bank.items()
    }
    if adapter_state.keys() != expected_state.keys():
        missing = sorted(expected_state.keys() - adapter_state.keys())
        extra = sorted(adapter_state.keys() - expected_state.keys())
        raise RuntimeError(
            "adapter checkpoint state is incomplete or has unknown tensors: "
            f"missing={missing[:4]} extra={extra[:4]}"
        )
    for name, value in adapter_state.items():
        expected = expected_state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError(f"adapter checkpoint tensor contract failed: {name}")
    training_contract = trained_extra.get("matchup_adapter_training_contract")
    if not isinstance(training_contract, dict) or (
        training_contract.get("schema")
        != "poke_bot.matchup_adapter_training_contract/v1"
        or training_contract.get("runtime_router_separate") is not True
        or training_contract.get("runtime_enabled") is not False
        or training_contract.get("routing")
        != "offline-oracle-package-and-full-deck-audited"
        or training_contract.get("optimizer_scope")
        != "matchup_adapter_bank_only"
        or list(training_contract.get("loss_scope") or [])
        != ["policy", "value"]
        or list(training_contract.get("expert_ids") or []) != list(EXPERT_IDS)
        or training_contract.get("adapter_config") != expected_config
    ):
        raise RuntimeError("adapter checkpoint training contract is missing or invalid")
    split_contract = dict(training_contract.get("split") or {})
    if (
        split_contract.get("schema")
        != "poke_bot.matchup_adapter_training_split/v1"
        or split_contract.get("routing")
        != "offline-oracle-package-and-full-deck-audited"
        or split_contract.get("runtime_router_separate") is not True
        or split_contract.get("corpus_manifest_digest")
        != training_contract.get("corpus_manifest_digest")
        or split_contract.get("active_gate_contract_digest")
        != training_contract.get("active_gate_contract_digest")
    ):
        raise RuntimeError("adapter checkpoint split contract is inconsistent")
    per_route = dict(split_contract.get("per_route") or {})
    if set(per_route) != set(EXPERT_IDS):
        raise RuntimeError("adapter training contract route roster is incomplete")

    trained_routes: list[str] = []
    zero_example_routes: list[str] = []
    invalid_route_coverage: list[str] = []
    for route, archetype_id in enumerate(EXPERT_IDS):
        counts = dict(per_route.get(archetype_id) or {})
        coverage = [
            int(counts.get(field, -1))
            for field in (
                "train_sequences",
                "train_decisions",
                "val_sequences",
                "val_decisions",
            )
        ]
        if any(value < 0 for value in coverage):
            invalid_route_coverage.append(archetype_id)
            continue
        up = [
            value
            for name, value in adapter_state.items()
            if name.startswith(f"matchup_adapter_bank.experts.{route}.up.")
        ]
        projection_is_nonzero = bool(up) and any(
            int(value.count_nonzero().item()) > 0 for value in up
        )
        if all(value == 0 for value in coverage):
            if projection_is_nonzero:
                invalid_route_coverage.append(archetype_id)
            else:
                zero_example_routes.append(archetype_id)
        elif all(value > 0 for value in coverage) and projection_is_nonzero:
            trained_routes.append(archetype_id)
        else:
            invalid_route_coverage.append(archetype_id)
    if invalid_route_coverage:
        raise RuntimeError(
            "adapter fit/coverage contract is inconsistent for routes: "
            f"{invalid_route_coverage}"
        )
    if (
        zero_example_routes
        and training_contract.get("zero_example_routes_remain_dormant") is not True
    ):
        raise RuntimeError(
            "adapter training contract does not authorize zero-example dormant routes"
        )
    try:
        for field_name, value in (
            (
                "training corpus manifest digest",
                training_contract.get("corpus_manifest_digest"),
            ),
            (
                "training active gate contract digest",
                training_contract.get("active_gate_contract_digest"),
            ),
            ("training membership digest", split_contract.get("membership_digest")),
        ):
            _require_digest(value, field_name=field_name)
        inputs = dict(training_contract.get("inputs") or {})
        if inputs.get("schema") != "poke_bot.matchup_adapter_input_provenance/v1":
            raise ValueError("input provenance schema mismatch")
        for field_name in (
            "source_jsonl_digest",
            "corpus_manifest_file_digest",
            "active_gate_contract_file_digest",
            "implementation_digest",
        ):
            _require_digest(
                inputs.get(field_name),
                field_name=f"training input {field_name}",
            )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("adapter checkpoint provenance digests are invalid") from exc
    validation = trained_extra.get("matchup_adapter_per_route_validation")
    if not isinstance(validation, dict):
        raise RuntimeError("adapter checkpoint lacks per-route validation evidence")
    invalid_validation = []
    for route, archetype_id in enumerate(EXPERT_IDS):
        row = dict(validation.get(archetype_id) or {})
        numeric = [
            float(row.get("total_loss", float("nan"))),
            float(row.get("policy_loss", float("nan"))),
            float(row.get("value_loss", float("nan"))),
        ]
        zero_example = archetype_id in zero_example_routes
        valid_zero = bool(
            zero_example
            and int(row.get("route", UNKNOWN_ROUTE)) == route
            and row.get("status") == "dormant_no_validation_examples"
            and int(row.get("n_games", -1)) == 0
            and int(row.get("n_decisions", -1)) == 0
        )
        valid_trained = bool(
            not zero_example
            and int(row.get("route", UNKNOWN_ROUTE)) == route
            and int(row.get("n_games", 0)) > 0
            and int(row.get("n_decisions", 0)) > 0
            and all(value == value and abs(value) != float("inf") for value in numeric)
        )
        if not (valid_zero or valid_trained):
            invalid_validation.append(archetype_id)
    if invalid_validation:
        raise RuntimeError(
            "adapter checkpoint has invalid per-route validation evidence: "
            f"{invalid_validation}"
        )

    # Deepcopy keeps every parent training object independent of the input
    # payload.  Only the model adapter keys and explicit dormant provenance are
    # allowed to differ.
    merged = copy.deepcopy(parent)
    merged_state = dict(merged.get("model_state_dict") or {})
    merged_state.update({name: value.detach().clone() for name, value in adapter_state.items()})
    merged["model_state_dict"] = merged_state
    merged_extra = dict(merged.get("extra") or {})
    merged_extra["matchup_adapter_config"] = copy.deepcopy(expected_config)
    merged_extra["matchup_adapter_training_contract"] = copy.deepcopy(
        training_contract
    )
    merged_extra["matchup_adapter_per_route_validation"] = copy.deepcopy(
        validation
    )
    merged_extra["dormant_matchup_adapter_bank"] = {
        "runtime_enabled": False,
        "activation_receipt": str(proof.path),
        "activation_parent_digest": proof.parent_checkpoint_digest,
        "activation_receipt_digest": receipt_digest,
        "adapter_checkpoint": str(adapter_path),
        "adapter_checkpoint_digest": _file_digest(adapter_path),
        "optimizer_imported": bool(import_optimizer_state),
    }
    phase_epochs = int(trained.get("epoch") or 0)
    phase_steps = int(trained.get("step") or 0)
    phase_route_sequences = {
        archetype_id: int(
            dict(per_route[archetype_id]).get("train_sequences", 0)
        )
        * phase_epochs
        for archetype_id in EXPERT_IDS
    }
    phase_route_decisions = {
        archetype_id: int(
            dict(per_route[archetype_id]).get("train_decisions", 0)
        )
        * phase_epochs
        for archetype_id in EXPERT_IDS
    }
    prior_fit = (
        dict(merged_extra.get("dormant_matchup_adapter_fit") or {})
        if bool(accumulate_parent_fit)
        else {}
    )
    if prior_fit and (
        prior_fit.get("schema") != "poke_bot.dormant_matchup_adapter_fit/v1"
        or prior_fit.get("optimizer_scope") != "matchup_adapter_bank_only"
        or prior_fit.get("base_frozen") is not True
    ):
        raise RuntimeError("parent dormant adapter fit contract is invalid")
    prior_route_sequences = {
        archetype_id: int(
            dict(prior_fit.get("route_sequences") or {}).get(archetype_id, 0)
        )
        for archetype_id in EXPERT_IDS
    }
    prior_route_decisions = {
        archetype_id: int(
            dict(prior_fit.get("route_decisions") or {}).get(archetype_id, 0)
        )
        for archetype_id in EXPERT_IDS
    }
    cumulative_route_sequences = {
        archetype_id: (
            prior_route_sequences[archetype_id]
            + phase_route_sequences[archetype_id]
        )
        for archetype_id in EXPERT_IDS
    }
    cumulative_route_decisions = {
        archetype_id: (
            prior_route_decisions[archetype_id]
            + phase_route_decisions[archetype_id]
        )
        for archetype_id in EXPERT_IDS
    }
    cumulative_trained_routes = [
        archetype_id
        for archetype_id in EXPERT_IDS
        if cumulative_route_decisions[archetype_id] > 0
    ]
    cumulative_dormant_routes = [
        archetype_id
        for archetype_id in EXPERT_IDS
        if cumulative_route_decisions[archetype_id] == 0
    ]
    merged_extra["dormant_matchup_adapter_fit"] = {
        "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
        "runtime_enabled": False,
        "base_frozen": True,
        "optimizer_scope": "matchup_adapter_bank_only",
        "optimizer_included": bool(import_optimizer_state),
        "optimizer_state_restored": bool(
            trained_extra.get(
                "matchup_adapter_parent_optimizer_state_restored", False
            )
        ),
        "activation_receipt": str(proof.path),
        "activation_receipt_digest": receipt_digest,
        "epochs": int(prior_fit.get("epochs") or 0) + phase_epochs,
        "steps": int(prior_fit.get("steps") or 0) + phase_steps,
        "rows": int(prior_fit.get("rows") or 0)
        + sum(phase_route_decisions.values()),
        "phase_epochs": phase_epochs,
        "phase_steps": phase_steps,
        "phase_rows": sum(phase_route_decisions.values()),
        "route_sequences": cumulative_route_sequences,
        "route_decisions": cumulative_route_decisions,
        "phase_route_sequences": phase_route_sequences,
        "phase_route_decisions": phase_route_decisions,
        "trained_archetype_ids": cumulative_trained_routes,
        "dormant_no_example_archetype_ids": cumulative_dormant_routes,
        "zero_example_routes_remain_dormant": True,
    }
    if bool(import_optimizer_state):
        optimizer_state = trained.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict) or not optimizer_state:
            raise RuntimeError(
                "adapter checkpoint lacks the isolated optimizer state requested "
                "for rehearsal continuation"
            )
        merged_extra["dormant_matchup_adapter_optimizer_state"] = copy.deepcopy(
            optimizer_state
        )
    merged["extra"] = merged_extra
    model_config = dict(merged.get("model_config") or {})
    if bool(model_config.get("matchup_adapters_enabled", False)):
        raise RuntimeError("parent model config unexpectedly activates adapters")
    # Do not add the optional false field to a legacy model profile: preserving
    # its structural config avoids an unrelated design-fingerprint migration.
    output = Path(output_path).expanduser().resolve()
    checkpoint_mod.immutable_torch_save(merged, output)
    return output


@dataclass(frozen=True)
class CorpusPackage:
    opponent_id: str
    content_digest: str
    archetype_id: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusManifest:
    digest: str
    packages_by_id: Mapping[str, CorpusPackage]
    packages_by_digest: Mapping[str, CorpusPackage]


def parse_corpus_manifest(payload: Mapping[str, Any]) -> CorpusManifest:
    """Validate exact package/archetype identities and alias collisions."""

    if str(payload.get("schema") or "") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError("invalid matchup-adapter corpus manifest schema")
    by_id: dict[str, CorpusPackage] = {}
    by_digest: dict[str, CorpusPackage] = {}
    for raw in payload.get("packages") or []:
        row = dict(raw)
        opponent_id = normalize_matchup_identity(row.get("opponent_id"))
        archetype_id = normalize_matchup_identity(row.get("archetype_id"))
        digest = _require_digest(
            row.get("content_digest"), field_name="corpus package digest"
        )
        aliases = tuple(
            normalize_matchup_identity(value) for value in row.get("aliases") or []
        )
        if not opponent_id or archetype_id not in EXPERT_IDS:
            raise ValueError("corpus package has an unsupported exact identity")
        if any(not alias for alias in aliases):
            raise ValueError("corpus package contains an empty alias")
        if len(set(aliases)) != len(aliases):
            raise ValueError("corpus package repeats an alias")
        if opponent_id in aliases:
            raise ValueError("corpus package aliases its canonical ID")
        package = CorpusPackage(opponent_id, digest, archetype_id, aliases)
        prior_digest = by_digest.get(digest)
        if prior_digest is not None and prior_digest != package:
            raise ValueError("one content digest maps to conflicting corpus packages")
        by_digest[digest] = package
        for identity in (opponent_id, *aliases):
            prior = by_id.get(identity)
            if prior is not None and prior != package:
                raise ValueError("corpus package alias collision")
            by_id[identity] = package
    if not by_digest:
        raise ValueError("corpus manifest has no packages")
    return CorpusManifest(
        digest=_canonical_json_digest(payload),
        packages_by_id=by_id,
        packages_by_digest=by_digest,
    )


@dataclass(frozen=True)
class GateExclusionSet:
    contract_digest: str
    opponent_ids: frozenset[str]
    content_digests: frozenset[str]


def gate_exclusions(payload: Mapping[str, Any]) -> GateExclusionSet:
    """Return package-deduplicated active-gate IDs, aliases, and digests."""

    next_gate = dict(payload.get("next_gate") or {})
    evaluation = dict(next_gate.get("evaluation") or {})
    if (
        str(payload.get("active_gate_id") or "") != str(next_gate.get("id") or "")
        or evaluation.get("formal_eval_disjoint_from_training") is not True
        or evaluation.get("package_digest_deduplicated") is not True
    ):
        raise ValueError("active gate lacks formal package-disjointness")
    ids: set[str] = set()
    digests: set[str] = set()
    for raw in next_gate.get("roster") or []:
        row = dict(raw)
        opponent_id = normalize_matchup_identity(row.get("opponent_id"))
        digest = _require_digest(
            row.get("content_digest"), field_name="active gate package digest"
        )
        if not opponent_id or opponent_id in ids or digest in digests:
            raise ValueError("active gate package identities are not unique")
        ids.add(opponent_id)
        digests.add(digest)
    for raw in next_gate.get("excluded_aliases") or []:
        row = dict(raw)
        alias = normalize_matchup_identity(row.get("opponent_id"))
        canonical = normalize_matchup_identity(row.get("canonical_opponent_id"))
        if not alias or canonical not in ids or alias in ids:
            raise ValueError("active gate alias contract is invalid")
        ids.add(alias)
    if not ids or not digests:
        raise ValueError("active gate exclusion set is empty")
    return GateExclusionSet(
        contract_digest=_canonical_json_digest(payload),
        opponent_ids=frozenset(ids),
        content_digests=frozenset(digests),
    )


@dataclass(frozen=True)
class AdapterTrainingTicket:
    schema: str
    opponent_id: str
    package_digest: str
    archetype_id: str
    route: int
    corpus_manifest_digest: str
    gate_contract_digest: str
    episode_id: str
    seat: int
    acting_archetype_id: str = "alakazam"


_TICKET_ATTRIBUTE = "_matchup_adapter_training_ticket_v1"


def _ticket_sequence(
    sequence: "GameSequence",
    *,
    package: CorpusPackage,
    corpus: CorpusManifest,
    gate: GateExclusionSet,
) -> None:
    ticket = AdapterTrainingTicket(
        schema=TRAINING_TICKET_SCHEMA,
        opponent_id=package.opponent_id,
        package_digest=package.content_digest,
        archetype_id=package.archetype_id,
        route=route_for_archetype(package.archetype_id),
        corpus_manifest_digest=corpus.digest,
        gate_contract_digest=gate.contract_digest,
        episode_id=str(sequence.episode_id),
        seat=int(sequence.seat),
        acting_archetype_id=normalize_matchup_identity(sequence.archetype),
    )
    payload = {
        "schema": ticket.schema,
        "opponent_id": ticket.opponent_id,
        "package_digest": ticket.package_digest,
        "archetype_id": ticket.archetype_id,
        "route": int(ticket.route),
        "corpus_manifest_digest": ticket.corpus_manifest_digest,
        "gate_contract_digest": ticket.gate_contract_digest,
        "episode_id": ticket.episode_id,
        "seat": int(ticket.seat),
        "acting_archetype_id": ticket.acting_archetype_id,
    }
    sequence.matchup_adapter_training_ticket = payload
    # Keep the legacy dynamic attribute while local cached objects may still
    # exist.  The explicit dataclass field above is the authoritative durable
    # representation.
    setattr(sequence, _TICKET_ATTRIBUTE, ticket)


def training_route_for_decision(
    sequence: "GameSequence",
    decision: "DecisionSample",
) -> int:
    """Return the offline oracle route after package/full-deck validation."""

    ticket = adapter_training_ticket(sequence)
    _validate_training_ticket_sequence(sequence, ticket)
    return _training_route_for_ticket(decision, ticket)


def training_routes_for_sequence(sequence: "GameSequence") -> tuple[int, ...]:
    """Validate one audited ticket and return every exact decision route.

    Router Format 6 resolves and validates the immutable slot registry while
    parsing the ticket.  Adapter training consumes complete sequences, so
    repeating that file-backed identity check for every decision is redundant
    and can dominate CPU time on million-row corpora.  This helper preserves
    the same fail-closed decision checks while pinning the validated ticket for
    exactly one sequence traversal.
    """

    ticket = adapter_training_ticket(sequence)
    _validate_training_ticket_sequence(sequence, ticket)
    return tuple(
        _training_route_for_ticket(decision, ticket)
        for decision in sequence.decisions
    )


def _validate_training_ticket_sequence(
    sequence: "GameSequence",
    ticket: AdapterTrainingTicket,
) -> None:
    """Fail closed if a parsed ticket no longer identifies its sequence."""

    raw_sequence_seat = getattr(sequence, "seat", None)
    if (
        normalize_matchup_identity(sequence.archetype)
        != ticket.acting_archetype_id
        or normalize_matchup_identity(sequence.opp_archetype) != ticket.archetype_id
        or str(sequence.episode_id) != ticket.episode_id
        or type(raw_sequence_seat) is not int
        or raw_sequence_seat not in (0, 1)
        or raw_sequence_seat != ticket.seat
        or training_route_for_archetype(ticket.archetype_id) != int(ticket.route)
    ):
        raise RuntimeError("matchup-adapter training ticket no longer matches sequence")


def _training_route_for_ticket(
    decision: "DecisionSample",
    ticket: AdapterTrainingTicket,
) -> int:
    """Validate one decision against an already validated sequence ticket."""

    raw_route = getattr(decision, "matchup_adapter_oracle_route", UNKNOWN_ROUTE)
    if type(raw_route) is not int:
        raise RuntimeError("oracle training route is not an exact integer identity")
    route = raw_route
    if route == UNKNOWN_ROUTE:
        raise RuntimeError("audited adapter sequence lost its oracle training route")
    if route != int(ticket.route):
        raise RuntimeError(
            "oracle training route contradicts the ground-truth matchup ticket"
        )
    raw_public_route = getattr(
        decision, "matchup_adapter_public_route", UNKNOWN_ROUTE
    )
    if type(raw_public_route) is not int:
        raise RuntimeError(
            "public-prefix audit route is not an exact integer identity"
        )
    public_route = raw_public_route
    if public_route not in (UNKNOWN_ROUTE, int(ticket.route)):
        raise RuntimeError(
            "public-prefix audit route contradicts the oracle matchup ticket"
        )
    return route


def adapter_training_ticket(sequence: "GameSequence") -> AdapterTrainingTicket:
    """Return the durable audited ticket or fail closed."""

    raw = dict(getattr(sequence, "matchup_adapter_training_ticket", None) or {})
    ticket: Any
    if raw:
        try:
            if isinstance(raw.get("route", UNKNOWN_ROUTE), bool) or isinstance(
                raw.get("seat", -1), bool
            ):
                raise ValueError("boolean route/seat is not an exact identity")
            ticket = AdapterTrainingTicket(
                schema=str(raw.get("schema") or ""),
                opponent_id=normalize_matchup_identity(raw.get("opponent_id")),
                package_digest=_require_digest(
                    raw.get("package_digest"), field_name="training ticket package digest"
                ),
                archetype_id=normalize_matchup_identity(raw.get("archetype_id")),
                route=int(raw.get("route", UNKNOWN_ROUTE)),
                corpus_manifest_digest=_require_digest(
                    raw.get("corpus_manifest_digest"),
                    field_name="training ticket corpus digest",
                ),
                gate_contract_digest=_require_digest(
                    raw.get("gate_contract_digest"),
                    field_name="training ticket gate digest",
                ),
                episode_id=str(raw.get("episode_id") or ""),
                seat=int(raw.get("seat", -1)),
                acting_archetype_id=normalize_matchup_identity(
                    raw.get("acting_archetype_id") or "alakazam"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("sequence has a malformed adapter training ticket") from exc
    else:
        ticket = getattr(sequence, _TICKET_ATTRIBUTE, None)
    if not isinstance(ticket, AdapterTrainingTicket) or ticket.schema != TRAINING_TICKET_SCHEMA:
        raise RuntimeError("sequence lacks an audited matchup-adapter training ticket")
    if (
        ticket.archetype_id not in training_route_target_ids()
        or ticket.acting_archetype_id not in training_route_target_ids()
        or training_route_for_archetype(ticket.archetype_id)
        != int(ticket.route)
        or not ticket.opponent_id
        or not ticket.episode_id
        or type(ticket.route) is not int
        or type(ticket.seat) is not int
        or int(ticket.seat) not in (0, 1)
    ):
        raise RuntimeError("sequence has an invalid adapter training ticket identity")
    return ticket


@dataclass
class PreparedAdapterCorpus:
    partitions: dict[str, list["GameSequence"]]
    excluded_gate_records: int = 0
    excluded_unsupported_records: int = 0
    duplicate_records: int = 0
    total_records: int = 0
    active_positions: int = 0
    runtime_recognized_positions: int = 0

    @property
    def sequences(self) -> list["GameSequence"]:
        return [
            sequence
            for archetype_id in EXPERT_IDS
            for sequence in self.partitions.get(archetype_id, [])
        ]


def assert_prepared_adapter_corpus_coverage(
    prepared: PreparedAdapterCorpus,
    *,
    minimum_episodes_per_route: int = 2,
) -> None:
    """Require every expert to have episode-disjoint train/validation material."""

    if int(minimum_episodes_per_route) < 2:
        raise ValueError("adapter coverage requires at least two episodes per route")
    failures: list[str] = []
    global_episode_routes: dict[tuple[str, int], str] = {}
    for archetype_id in EXPERT_IDS:
        sequences = list(prepared.partitions.get(archetype_id) or [])
        episode_keys = {(str(row.episode_id), int(row.seat)) for row in sequences}
        active = sum(
            int(
                getattr(decision, "matchup_adapter_oracle_route", UNKNOWN_ROUTE)
                != UNKNOWN_ROUTE
            )
            for row in sequences
            for decision in row.decisions
        )
        if len(episode_keys) < int(minimum_episodes_per_route) or active <= 0:
            failures.append(
                f"{archetype_id}:episodes={len(episode_keys)} active={active}"
            )
        for key in episode_keys:
            prior = global_episode_routes.setdefault(key, archetype_id)
            if prior != archetype_id:
                failures.append(
                    f"episode_route_collision:{key[0]}:seat{key[1]}:{prior}/{archetype_id}"
                )
    if failures:
        raise RuntimeError(
            "matchup-adapter corpus coverage failed: " + "; ".join(failures)
        )


def prepare_adapter_corpus_records(
    records: Iterable[Mapping[str, Any]],
    *,
    corpus_manifest: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
    max_context: int = 320,
) -> PreparedAdapterCorpus:
    """Split audited raw histories into oracle training partitions.

    Supported records with missing/ambiguous identity or package provenance are
    rejected, not guessed.  Active-gate IDs, aliases, and content digests are
    filtered before conversion.  Duplicate package/episode/seat rows are kept
    once.  Only Alakazam acting-seat records are eligible.
    """

    from .dataset import convert_record

    corpus = parse_corpus_manifest(corpus_manifest)
    gate = gate_exclusions(gate_contract)
    prepared = PreparedAdapterCorpus(partitions={value: [] for value in EXPERT_IDS})
    seen: set[tuple[str, str, int]] = set()
    episode_routes: dict[tuple[str, int], str] = {}
    for raw_record in records:
        prepared.total_records += 1
        record = dict(raw_record)
        if normalize_matchup_identity(record.get("archetype")) != "alakazam":
            prepared.excluded_unsupported_records += 1
            continue
        episode_id = str(record.get("episode_id") or "").strip()
        raw_seat = record.get("seat")
        if (
            not episode_id
            or isinstance(raw_seat, bool)
            or not isinstance(raw_seat, int)
            or raw_seat not in (0, 1)
        ):
            raise ValueError("record lacks an exact episode/acting-seat identity")
        provenance = dict(record.get("target_provenance") or {})
        if (
            provenance.get("formal_eval") is True
            or normalize_matchup_identity(provenance.get("opponent_training_group"))
            == "formal_eval"
            or record.get("training_eligible") is False
        ):
            prepared.excluded_gate_records += 1
            continue
        opponent_id = normalize_matchup_identity(
            provenance.get("opponent_id")
            or provenance.get("opponent_package_id")
        )
        digest = _require_digest(
            provenance.get("opponent_content_digest")
            or provenance.get("opponent_package_digest"),
            field_name="record opponent package digest",
        )
        package_by_id = corpus.packages_by_id.get(opponent_id)
        package_by_digest = corpus.packages_by_digest.get(digest)
        if (
            package_by_id is None
            or package_by_digest is None
            or package_by_id != package_by_digest
        ):
            raise ValueError("record package ID/digest is unknown or ambiguous")
        package = package_by_id
        if opponent_id in gate.opponent_ids or digest in gate.content_digests:
            prepared.excluded_gate_records += 1
            continue
        expected_archetype = package.archetype_id
        recorded_label = normalize_matchup_identity(record.get("opp_archetype"))
        allowed_package_labels = {
            package.opponent_id,
            *package.aliases,
        }
        if recorded_label in EXPERT_IDS:
            if recorded_label != expected_archetype:
                raise ValueError(
                    "record package and ground-truth archetype disagree"
                )
        elif recorded_label not in allowed_package_labels:
            raise ValueError(
                "record opponent label is neither its exact package nor archetype"
            )
        dedup_key = (digest, episode_id, raw_seat)
        if dedup_key in seen:
            prepared.duplicate_records += 1
            continue
        seen.add(dedup_key)
        episode_key = (episode_id, raw_seat)
        prior_route = episode_routes.setdefault(episode_key, expected_archetype)
        if prior_route != expected_archetype:
            raise ValueError("one episode/seat appears in conflicting route partitions")

        # Route the complete public prefix before applying the model's context
        # cap.  This preserves recognition state at a 320-row packing boundary
        # without letting a future state change an earlier mask.
        router = MatchupAdapterGameRouter(first_eligible_iteration=FIRST_ELIGIBLE_ITERATION)
        all_routes = [
            int(router.observe(dict(step).get("observation") or {}).route)
            for step in list(record.get("steps") or [])
        ]
        sequence, reason, _details = convert_record(
            record,
            max_context=int(max_context),
            verify_info_set=True,
        )
        if sequence is None:
            raise ValueError(f"record failed exact-history conversion: {reason}")
        # The package manifest is authoritative only after its ID+digest pair
        # validated above.  Normalize public-mix records that originally carry
        # the package ID in ``opp_archetype`` to their audited deck family.
        sequence.opp_archetype = expected_archetype
        aligned_routes = all_routes[-len(sequence.decisions) :]
        if len(aligned_routes) != len(sequence.decisions):
            raise AssertionError("public route mask lost sequence alignment")
        expected_route = route_for_archetype(expected_archetype)
        for decision, public_route in zip(sequence.decisions, aligned_routes):
            if public_route not in (UNKNOWN_ROUTE, expected_route):
                raise ValueError(
                    "public recognizer contradicts corpus matchup; possible label "
                    "collision or contaminated history"
                )
            decision.matchup_adapter_oracle_route = int(expected_route)
            decision.matchup_adapter_public_route = int(public_route)
        _ticket_sequence(sequence, package=package, corpus=corpus, gate=gate)
        prepared.active_positions += len(sequence.decisions)
        prepared.runtime_recognized_positions += sum(
            int(route != UNKNOWN_ROUTE) for route in aligned_routes
        )
        prepared.partitions[expected_archetype].append(sequence)
    return prepared


__all__ = [
    "ACTIVATION_AFTER_COMPLETED_ITERATION",
    "ACTIVATION_RECEIPT_SCHEMA",
    "ADAPTER_REHEARSAL_AUTHORIZATION_SCHEMA",
    "SPECIALIST_BOOTSTRAP_AUTHORIZATION_SCHEMA",
    "AUDITED_RUNTIME_EXPERT_IDS",
    "AUDITED_RUNTIME_ROUTES",
    "AdapterTrainingTicket",
    "ActivationReceipt",
    "CORPUS_MANIFEST_SCHEMA",
    "FIRST_ELIGIBLE_ITERATION",
    "GateExclusionSet",
    "MatchupAdapterGameRouter",
    "MatchupAdapterShadowAudit",
    "PUBLIC_SIGNATURE_GROUPS",
    "UNROUTABLE_EXPERT_IDS",
    "UNROUTABLE_PUBLIC_CONFLICT_IDS",
    "PreparedAdapterCorpus",
    "PublicMatchupDecision",
    "PublicMatchupRecognizer",
    "ShadowMatchupAdapterRouter",
    "UNACCEPTED_RUNTIME_ROUTES",
    "build_activation_receipt",
    "build_adapter_rehearsal_authorization",
    "assert_prepared_adapter_corpus_coverage",
    "adapter_training_ticket",
    "training_route_for_archetype",
    "training_route_target_ids",
    "gate_exclusions",
    "merge_dormant_adapter_checkpoint",
    "normalize_matchup_identity",
    "parse_corpus_manifest",
    "prepare_adapter_corpus_records",
    "runtime_model_route",
    "training_route_for_decision",
    "training_routes_for_sequence",
    "validate_activation_receipt",
    "validate_adapter_training_authorization",
]
