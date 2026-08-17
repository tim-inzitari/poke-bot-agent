"""Trusted public-history particle MCTS (POMCP-style root sampling).

Every simulation samples a fresh hidden-state particle, starts a fresh libcg
search world, and updates one shared action-observation history tree. The tree
is keyed by acting-player information states, never by particle identity. This
aggregates root actions across the belief without per-determinization action
selection (strategy fusion).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import torch

from . import cg_env, config, features
from .batched_infer import LeafPacket, forward_leaf_batch
from .belief import (
    EmpiricalDeckPosterior,
    HiddenStateParticle,
    NeuralBeliefPriors,
    PublicBeliefHistory,
    assert_deployment_observation,
    simulator_version,
)
from .blackwell_heads import (
    blackwell_strategy_heads_enabled,
    root_value_bias_from_lethal,
)
from .matchup_adapter_activation import ShadowMatchupAdapterRouter
from .mcts import GameClock, MCTSResult
from .model import TemporalCabtTransformer, card_prior_logits_or_uniform
from .replay_import import assert_info_set as assert_public_info_set
from .search_targets import build_search_target, select_by_visits


@dataclass
class BeliefEdge:
    action: list[int]
    prior: float
    visit: int = 0
    total: float = 0.0
    outcomes: dict[str, "BeliefNode"] = field(default_factory=dict)
    # A public-history root-sampled tree may not treat a sampled/opaque chance
    # edge as a reusable deterministic real-game continuation.
    saw_sampled_or_opaque_chance: bool = False

    def q(self) -> float:
        return self.total / self.visit if self.visit else 0.0


class TrustedSearchBudgetExhausted(RuntimeError):
    """A move deadline expired before the minimum trusted simulation contract."""


@dataclass
class BeliefNode:
    fingerprint: str
    actor: int
    depth: int
    edges: list[BeliefEdge] = field(default_factory=list)
    visit: int = 0
    total: float = 0.0
    network_evaluated: bool = False
    factorized: bool = False
    action_prefix: tuple[int, ...] = ()
    total_action_count: int = 0
    # A node's bootstrap value is deliberately separate from its backed-up Q.
    # Exact finite-chance expansion can visit every forced child in one parent
    # simulation; a newly created child has no Q yet, but it does have the
    # frozen leaf value produced when its legal decision was evaluated.
    bootstrap_value: Optional[float] = None

    def q(self) -> float:
        return self.total / self.visit if self.visit else 0.0


@dataclass
class _BranchHistory:
    boards: dict[int, list[features.SparseVector]]
    previous_actions: dict[int, list[Optional[features.SparseVector]]]
    last_action: dict[int, Optional[features.SparseVector]]


class ExactFiniteChanceUnavailable(RuntimeError):
    """A capability cannot force this particular chance node safely.

    This is an expected, fail-closed condition.  The caller retains the
    pre-random leaf boundary and labels the result accordingly; it must never
    infer an exact expectation from a partial capability or sample a private
    coin/die outcome.
    """


@dataclass(frozen=True)
class ExactFiniteChanceOutcome:
    """One force-enumerated, independently advanceable chance successor.

    ``probability`` is intentionally an exact :class:`Fraction`, not a float.
    The opaque ``successor`` may be a libcg search state or a backend-specific
    handle, but it must expose an ``observation`` usable by the search.  The
    two receipts are supplied by the capability: one binds the forced child
    and one attests that its future legal decision is available to inspect.
    """

    label: str
    probability: Fraction
    successor: Any
    successor_receipt: str
    future_legality_receipt: str


@dataclass(frozen=True)
class ExactFiniteChanceExpansion:
    """The complete sealed distribution returned by an optional backend.

    A backend must return every outcome before the MCTS is allowed to use an
    expected value.  ``force_enumeration_receipt`` and
    ``probability_receipt`` deliberately make that attestation explicit in
    the capability surface instead of treating a sampled manual-coin select
    as if it were an enumerable distribution.
    """

    outcomes: tuple[ExactFiniteChanceOutcome, ...]
    force_enumeration_receipt: str
    probability_receipt: str


@runtime_checkable
class ExactFiniteChanceCapability(Protocol):
    """Optional non-mutating force-enumeration adapter for simple chance.

    Returning ``None`` (or raising :class:`ExactFiniteChanceUnavailable`)
    means the backend cannot safely enumerate this node.  The method must not
    advance or mutate ``search_state``; each returned child must be separately
    advanceable through its opaque successor capability.  A nonterminal child
    either exposes a ``searchId`` accepted by ``cg_env.search_step`` or the
    capability supplies ``advance_exact_finite_chance_successor(child, action)``.
    """

    def enumerate_exact_finite_chance(
        self,
        search_state: Any,
    ) -> Optional[ExactFiniteChanceExpansion]:
        ...


@dataclass
class _FiniteChanceRollout:
    """One fully completed descendant rollout or exact chance expectation."""

    value: float
    terminal_results_seen: int = 0
    leaf_evaluations: int = 0
    simulator_transitions: int = 0
    # Logical path layers, not the sum of fan-out transitions.  A chance node
    # is one layer even when its complete distribution has six children.
    max_simulator_steps: int = 0
    finite_chance_enumerations: int = 0
    finite_chance_outcomes_enumerated: int = 0
    finite_chance_weighted_backup_count: int = 0
    finite_chance_forced_successor_transitions: int = 0
    # An unresolved random event is a leaf boundary, never a sampled private
    # continuation.  These counters are kept separate from the legacy sampled
    # fields below so receipts can prove zero guessed chance transitions.
    unforceable_chance_boundary_nodes: int = 0
    unforceable_chance_boundary_leaf_evaluations: int = 0
    unforceable_chance_boundary_reasons: dict[str, int] = field(
        default_factory=dict
    )
    chance_samples: int = 0
    sampled_unforceable_chance_nodes: int = 0
    sampled_unforceable_chance_reasons: dict[str, int] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _CachedLeafEvaluation:
    """Immutable model output reused only inside one real-decision tree.

    Tree statistics deliberately remain on the individual :class:`BeliefNode`.
    Reusing a model forward must not make unrelated transpositions share visits
    or value totals.  The cache therefore owns only the legal action ordering,
    priors and leaf value that came from the frozen model.
    """

    combos: tuple[tuple[int, ...], ...]
    priors: tuple[float, ...]
    value: float
    total_action_count: int
    factorized: bool
    action_prefix: tuple[int, ...]


@dataclass
class _DecisionEvaluationCache:
    """Per-search-call cache for exact model-input-equivalent leaf forwards.

    It is intentionally allocated inside ``_search_impl`` and never placed on
    ``BeliefMCTS``.  In particular, no entry can survive to another real action
    or turn.  A key includes the public observation, model-visible history,
    complete legal ordering, selected factorized prefix, acting deck and the
    hidden/chance scenario identity so a distinct sampled world never borrows a
    value from another world merely because their public projections coincide.
    """

    entries: dict[str, _CachedLeafEvaluation] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    deterministic_state_keys: set[str] = field(default_factory=set)
    hidden_or_chance_partitioned_keys: set[str] = field(default_factory=set)


def _raw_observation(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if dataclasses.is_dataclass(obs):
        return dataclasses.asdict(obs)
    raise TypeError(f"unsupported observation type {type(obs).__name__}")


def information_state_fingerprint(obs: Any) -> str:
    """Hash only the acting player's deployment-visible information state."""
    raw = _raw_observation(obs)
    assert_public_info_set(raw)
    projection = {
        "select": raw.get("select"),
        "logs": raw.get("logs"),
        "current": raw.get("current"),
    }
    encoded = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sparse_vector_payload(
    vector: Optional[features.SparseVector],
) -> Optional[dict[str, Any]]:
    """Return the exact serializable part of a model history token."""

    if vector is None:
        return None
    return {
        "index": [int(value) for value in vector.index],
        "value": [float(value) for value in vector.value],
        "offset": [int(value) for value in vector.offset],
        "pos": int(vector.pos),
    }


def _hidden_particle_identity(particle: Optional[HiddenStateParticle]) -> str:
    """Identify a root-sampled hidden world without exposing it to the model."""

    if particle is None:
        return "public-root"
    encoded = json.dumps(
        particle.search_inputs,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return "particle:" + hashlib.sha256(encoded).hexdigest()


def is_explicit_chance(obs: Any) -> bool:
    """The competition API exposes coin outcomes as YES/NO when manual_coin."""
    raw = _raw_observation(obs)
    select = raw.get("select") or {}
    return int(select.get("context", -1)) == 46  # SelectContext.COIN_HEAD


def factorize_visit_policy(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    policy: Sequence[float],
    selected: Sequence[int],
) -> list[dict[str, Any]]:
    """Project a complete action policy onto ordered autoregressive stages."""
    if len(action_combos) != len(policy):
        raise ValueError("action/policy length mismatch")
    full = [(list(combo), float(weight)) for combo, weight in zip(action_combos, policy)]
    selected_list = list(selected)
    prefix: list[int] = []
    rows: list[dict[str, Any]] = []
    while True:
        candidates = features.factorized_action_candidates(obs, prefix)
        if len(candidates) == 1 and candidates[0] == prefix:
            break
        weights = [0.0] * len(candidates)
        candidate_index = {tuple(candidate): i for i, candidate in enumerate(candidates)}
        for combo, weight in full:
            if combo[: len(prefix)] != prefix:
                continue
            key = tuple(prefix if len(combo) == len(prefix) else combo[: len(prefix) + 1])
            if key in candidate_index:
                weights[candidate_index[key]] += weight
        total = sum(weights)
        if total <= 0:
            raise ValueError("visit policy has zero mass at selected prefix")
        weights = [weight / total for weight in weights]
        next_selected = (
            list(prefix)
            if len(selected_list) == len(prefix)
            else selected_list[: len(prefix) + 1]
        )
        if tuple(next_selected) not in candidate_index:
            raise ValueError("selected action is absent from factorized candidates")
        rows.append(
            {
                "action_combos": [list(candidate) for candidate in candidates],
                "policy": weights,
                "selected_index": candidate_index[tuple(next_selected)],
            }
        )
        if next_selected == prefix:
            break
        prefix = next_selected
    return rows


class BeliefMCTS:
    """Root-sampled information-set MCTS with sequential adaptive updates."""

    def __init__(
        self,
        model: Optional[TemporalCabtTransformer],
        own_deck: Sequence[int],
        posterior: EmpiricalDeckPosterior,
        *,
        checkpoint_digest: str,
        model_generation: int,
        device: Optional[torch.device] = None,
        leaf_backend: Optional[Callable] = None,
        puct_c: Optional[float] = None,
        rng: Optional[random.Random] = None,
        min_trusted_sims: int = 128,
        particle_count: int = 16,
        max_depth: int = 1_000_000,
        convergence_min_sims: int = 32,
        convergence_stable_sims: int = 16,
        convergence_visit_share: float = 0.90,
        convergence_q_margin: float = 0.05,
        max_context: Optional[int] = None,
        matchup_shadow_router: Optional[ShadowMatchupAdapterRouter] = None,
        matchup_model_route: int = -1,
        search_backend: Optional[cg_env.SearchBackend] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
        exact_finite_chance_capability: Optional[ExactFiniteChanceCapability] = None,
        # ``finite_chance_capability`` is retained as a small spelling alias
        # for experimental callers while the explicit name remains canonical.
        finite_chance_capability: Optional[ExactFiniteChanceCapability] = None,
    ) -> None:
        if not checkpoint_digest.startswith("sha256:"):
            raise ValueError("trusted search requires an immutable sha256 checkpoint")
        if int(model_generation) < 0:
            raise ValueError("model generation must be non-negative")
        self.model = model
        self.own_deck = tuple(int(card) for card in own_deck)
        if len(self.own_deck) != 60:
            raise ValueError("own deck must contain exactly 60 cards")
        self.posterior = posterior
        self.checkpoint_digest = checkpoint_digest
        self.model_generation = int(model_generation)
        self.device = (
            device
            if device is not None
            else (
                next(model.parameters()).device
                if model is not None
                else torch.device("cpu")
            )
        )
        if leaf_backend is None:
            if model is None:
                raise ValueError("belief MCTS requires a model or leaf backend")
            self.leaf_eval = lambda packets: forward_leaf_batch(model, packets)
            self.leaf_evaluator_source = (
                "trained_checkpoint_policy_value_head"
            )
            self.leaf_evaluator_checkpoint_digest = checkpoint_digest
        else:
            self.leaf_eval = leaf_backend
            self.leaf_evaluator_source = str(
                getattr(leaf_backend, "source", "external_leaf_backend")
            )
            self.leaf_evaluator_checkpoint_digest = getattr(
                leaf_backend, "checkpoint_digest", None
            )
            if (
                self.leaf_evaluator_source
                == "trained_checkpoint_policy_value_head"
                and self.leaf_evaluator_checkpoint_digest != checkpoint_digest
            ):
                raise ValueError(
                    "trained leaf evaluator checkpoint digest does not match "
                    "the searched policy checkpoint"
                )
        self.puct_c = float(puct_c if puct_c is not None else config.SEARCH.puct_c)
        self.rng = rng or random.Random()
        self.min_trusted_sims = int(min_trusted_sims)
        if self.min_trusted_sims < 1:
            raise ValueError("min_trusted_sims must be at least one completed backup")
        self.particle_count = max(2, int(particle_count))
        self.max_depth = int(max_depth)
        self.convergence_min_sims = max(1, int(convergence_min_sims))
        self.convergence_stable_sims = max(1, int(convergence_stable_sims))
        self.convergence_visit_share = float(convergence_visit_share)
        self.convergence_q_margin = float(convergence_q_margin)
        if not 0.5 <= self.convergence_visit_share <= 1.0:
            raise ValueError("convergence_visit_share must be in [0.5, 1.0]")
        if self.convergence_q_margin < 0.0:
            raise ValueError("convergence_q_margin must be non-negative")
        self.max_context = int(
            max_context
            if max_context is not None
            else getattr(model, "max_context", config.MODEL.max_context)
        )
        self._simulator_version = simulator_version()
        self.matchup_shadow_router = matchup_shadow_router
        if type(matchup_model_route) is not int:
            raise TypeError("matchup_model_route must be an exact integer")
        self.matchup_model_route = matchup_model_route
        self.search_backend: cg_env.SearchBackend = (
            search_backend if search_backend is not None else cg_env
        )
        self.stop_requested = stop_requested or (lambda: False)
        if (
            exact_finite_chance_capability is not None
            and finite_chance_capability is not None
            and exact_finite_chance_capability is not finite_chance_capability
        ):
            raise ValueError(
                "pass only one exact finite-chance capability spelling"
            )
        self.exact_finite_chance_capability = (
            exact_finite_chance_capability
            if exact_finite_chance_capability is not None
            else finite_chance_capability
        )
        # Keep a concrete alias on the instance too, so external experimental
        # bindings can discover the optional capability without guessing its
        # canonical constructor spelling.
        self.finite_chance_capability = self.exact_finite_chance_capability

    def _telemetry_mark(self):
        marker = getattr(self.leaf_eval, "telemetry_mark", None)
        return marker() if callable(marker) else None

    def _root_neural_priors(
        self,
        *,
        root_history_boards: Sequence[features.SparseVector],
        root_history_previous_actions: Sequence[
            Optional[features.SparseVector]
        ],
    ) -> NeuralBeliefPriors:
        """Root-only belief / Scope-B strategy priors; never into board tokens.

        Warm-started / missing Scope A card heads force uniform particle
        fallback until trained. Scope B lethal/prize-race are included only
        when ``blackwell_strategy_heads_enabled`` (Hammer Blackwell). Leaf
        servers stay policy+value only.
        """
        model = self.model
        if model is None or not root_history_boards:
            return NeuralBeliefPriors(uniform_fallback=True)
        warm = tuple(getattr(model, "warm_started_belief_heads", ()) or ())
        card_uniform = bool(
            "opp_hand_head" in warm or "opp_remainder_head" in warm
        )
        scope_b = blackwell_strategy_heads_enabled()
        # Fresh warm-start of Scope B heads → do not bias search yet.
        scope_b_ready = scope_b and not (
            "lethal_threat_head" in warm or "prize_race_head" in warm
        )
        try:
            with torch.inference_mode():
                # encode_history → state_vec only (no options / no board rewrite).
                state, _spatial = model.encode_history(
                    list(root_history_boards),
                    previous_actions=list(root_history_previous_actions),
                )
                belief = model.belief_aux_logits(state)
            aux = belief.get("aux_logits")
            hand = belief.get("opp_hand_logits")
            rem = belief.get("opp_remainder_logits")
            arch = (
                tuple(float(x) for x in aux[0].detach().cpu().tolist())
                if aux is not None
                else None
            )
            lethal_f: Optional[float] = None
            race_t: Optional[tuple[float, float]] = None
            if scope_b_ready:
                lethal = belief.get("lethal_threat_logits")
                race = belief.get("prize_race_pred")
                if lethal is not None:
                    lethal_f = float(lethal.reshape(-1)[0].detach().cpu().item())
                if race is not None and race.numel() >= 2:
                    row = race.reshape(-1, 2)[0].detach().cpu().tolist()
                    race_t = (float(row[0]), float(row[1]))
            if card_uniform:
                return NeuralBeliefPriors(
                    archetype_logits=arch,
                    opp_hand_logits=None,
                    opp_remainder_logits=None,
                    lethal_threat_logit=lethal_f,
                    prize_race=race_t,
                    uniform_fallback=False,
                )
            card_vocab = int(getattr(model, "belief_card_vocab", 0) or 0)
            hand_t = card_prior_logits_or_uniform(
                hand[0] if hand is not None else None,
                card_vocab or (int(hand.shape[-1]) if hand is not None else 0),
            )
            rem_t = card_prior_logits_or_uniform(
                rem[0] if rem is not None else None,
                card_vocab or (int(rem.shape[-1]) if rem is not None else 0),
            )
            return NeuralBeliefPriors(
                archetype_logits=arch,
                opp_hand_logits=tuple(
                    float(x) for x in hand_t.detach().cpu().tolist()
                ),
                opp_remainder_logits=tuple(
                    float(x) for x in rem_t.detach().cpu().tolist()
                ),
                lethal_threat_logit=lethal_f,
                prize_race=race_t,
                uniform_fallback=False,
            )
        except Exception:
            return NeuralBeliefPriors(uniform_fallback=True)

    def _telemetry_since(self, marker) -> dict[str, Any]:
        summary = getattr(self.leaf_eval, "telemetry_since", None)
        if marker is not None and callable(summary):
            return dict(summary(marker))
        return {
            "remote_requests": 0,
            "remote_leaves": 0,
            "queue_wait_ms_mean": 0.0,
            "queue_wait_ms_p95": 0.0,
            "inference_batch_size_mean": 1.0,
            "inference_batch_size_p95": 1.0,
            "server_inference_ms_mean": 0.0,
            "client_roundtrip_ms_mean": 0.0,
        }

    @staticmethod
    def _terminal_value(obs: Any, root_seat: int) -> Optional[float]:
        raw = _raw_observation(obs)
        current = raw.get("current")
        if not isinstance(current, dict):
            return None
        result = int(current.get("result", -1))
        if result < 0:
            return None
        return 0.0 if result == 2 else (1.0 if result == root_seat else -1.0)

    @staticmethod
    def _actor(obs: Any) -> int:
        raw = _raw_observation(obs)
        current = raw.get("current") or {}
        actor = int(current.get("yourIndex", -1))
        if actor not in (0, 1):
            raise ValueError("search observation has invalid actor")
        return actor

    def _packet(
        self,
        obs: Any,
        *,
        root_seat: int,
        particle: Optional[HiddenStateParticle],
        branch: _BranchHistory,
        combos: list[list[int]],
    ) -> LeafPacket:
        raw = _raw_observation(obs)
        actor = self._actor(raw)
        deck = (
            list(self.own_deck)
            if actor == root_seat
            else list(particle.opponent_deck if particle is not None else ())
        )
        if len(deck) != 60:
            raise ValueError("leaf actor deck particle is missing")
        return LeafPacket(
            obs=raw,
            your_deck=deck,
            root_seat=root_seat,
            history_boards=list(branch.boards[actor]),
            history_previous_actions=list(branch.previous_actions[actor]),
            action_combos_override=combos,
            # Older reconstructed/test engines predate the matchup-adapter
            # route field.  Their trusted behavior is the same exact dormant
            # bypass used by a normally constructed engine's default.
            matchup_route=getattr(self, "matchup_model_route", -1),
        )

    def _record_observation(
        self,
        obs: Any,
        *,
        root_seat: int,
        particle: HiddenStateParticle,
        branch: _BranchHistory,
    ) -> None:
        raw = _raw_observation(obs)
        features.assert_info_set(raw)
        actor = self._actor(raw)
        deck = list(self.own_deck if actor == root_seat else particle.opponent_deck)
        branch.boards[actor].append(features.build_board_tokens(raw, deck))
        branch.previous_actions[actor].append(branch.last_action[actor])
        max_context = self.max_context
        branch.boards[actor] = branch.boards[actor][-max_context:]
        branch.previous_actions[actor] = branch.previous_actions[actor][-max_context:]

    def _leaf_evaluation_key(
        self,
        packet: LeafPacket,
        *,
        action_prefix: Sequence[int],
        hidden_or_chance_scenario: str,
    ) -> str:
        """Hash every frozen-model input that can change a leaf result.

        The information-state projection alone is deliberately insufficient:
        the actor's exact history and root-sampled hidden world can change the
        frozen forward packet even when the public board happens to look the
        same.  Conversely, a repeat of this full key is a genuine deterministic
        repeat and must not issue another model forward in the same tree.
        """

        payload = {
            "schema": "poke_bot.belief_mcts_leaf_evaluation_cache/v1",
            "information_state_fingerprint": information_state_fingerprint(
                packet.obs
            ),
            "actor": self._actor(packet.obs),
            "root_seat": int(packet.root_seat),
            "acting_deck": [int(card) for card in packet.your_deck],
            "history_boards": [
                _sparse_vector_payload(vector)
                for vector in (packet.history_boards or [])
            ],
            "history_previous_actions": [
                _sparse_vector_payload(vector)
                for vector in (packet.history_previous_actions or [])
            ],
            "complete_or_factorized_legal_actions": [
                [int(index) for index in combo]
                for combo in (packet.action_combos_override or [])
            ],
            "matchup_route": int(packet.matchup_route),
            "action_prefix": [int(index) for index in action_prefix],
            "hidden_or_chance_scenario": str(hidden_or_chance_scenario),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _apply_cached_leaf(
        node: BeliefNode,
        cached: _CachedLeafEvaluation,
        *,
        combos: Sequence[Sequence[int]],
        factorized: bool,
        action_prefix: Sequence[int],
        total_action_count: int,
    ) -> float:
        """Copy immutable model output into a node with fresh tree statistics."""

        expected_combos = tuple(tuple(int(index) for index in combo) for combo in combos)
        if cached.combos != expected_combos:
            raise RuntimeError("cached leaf legal action ordering does not match node")
        if cached.factorized != bool(factorized):
            raise RuntimeError("cached leaf factorization mode does not match node")
        if cached.action_prefix != tuple(int(index) for index in action_prefix):
            raise RuntimeError("cached leaf action prefix does not match node")
        if cached.total_action_count != int(total_action_count):
            raise RuntimeError("cached leaf action count does not match node")
        node.total_action_count = cached.total_action_count
        node.factorized = cached.factorized
        node.action_prefix = cached.action_prefix
        node.edges = [
            BeliefEdge(action=list(combo), prior=float(prior))
            for combo, prior in zip(cached.combos, cached.priors)
        ]
        node.network_evaluated = True
        node.bootstrap_value = float(cached.value)
        return node.bootstrap_value

    def _evaluate_node_cached(
        self,
        node: BeliefNode,
        obs: Any,
        *,
        root_seat: int,
        particle: Optional[HiddenStateParticle],
        branch: _BranchHistory,
        action_prefix: Sequence[int] = (),
        evaluation_cache: Optional[_DecisionEvaluationCache] = None,
        hidden_or_chance_scenario: Optional[str] = None,
    ) -> tuple[float, bool]:
        """Evaluate one leaf, returning ``(value, issued_model_forward)``.

        The optional cache is intentionally a call-local object owned by
        ``_search_impl``.  Callers outside a search keep the old one-shot
        behavior by passing no cache.
        """

        raw = _raw_observation(obs)
        prefix = [int(index) for index in action_prefix]
        total_action_count = features.ordered_action_count(raw)
        factorized = False
        if prefix:
            combos = features.factorized_action_candidates(raw, prefix)
            factorized = True
        else:
            try:
                combos = features.enumerate_action_combos(raw)
            except features.ActionSpaceTooLarge:
                combos = features.factorized_action_candidates(raw, [])
                factorized = True
        packet = self._packet(
            obs,
            root_seat=root_seat,
            particle=particle,
            branch=branch,
            combos=[list(combo) for combo in combos],
        )
        scenario = (
            str(hidden_or_chance_scenario)
            if hidden_or_chance_scenario is not None
            else _hidden_particle_identity(particle)
        )
        cache_key: Optional[str] = None
        if evaluation_cache is not None:
            cache_key = self._leaf_evaluation_key(
                packet,
                action_prefix=prefix,
                hidden_or_chance_scenario=scenario,
            )
            cached = evaluation_cache.entries.get(cache_key)
            if cached is not None:
                evaluation_cache.hits += 1
                return (
                    self._apply_cached_leaf(
                        node,
                        cached,
                        combos=combos,
                        factorized=factorized,
                        action_prefix=prefix,
                        total_action_count=total_action_count,
                    ),
                    False,
                )

        output = self.leaf_eval([packet])
        if len(output) != 1:
            raise RuntimeError("leaf backend returned wrong response count")
        evaluated = output[0]
        if evaluated.combos != list(combos):
            raise RuntimeError("leaf backend changed complete legal action ordering")
        if len(evaluated.priors) != len(combos):
            raise RuntimeError("leaf prior/action count mismatch")
        cached = _CachedLeafEvaluation(
            combos=tuple(tuple(int(index) for index in combo) for combo in combos),
            priors=tuple(float(prior) for prior in evaluated.priors),
            value=float(evaluated.value),
            total_action_count=int(total_action_count),
            factorized=bool(factorized),
            action_prefix=tuple(prefix),
        )
        if evaluation_cache is not None and cache_key is not None:
            evaluation_cache.entries[cache_key] = cached
            evaluation_cache.misses += 1
            evaluation_cache.deterministic_state_keys.add(cache_key)
            if scenario != "public-root":
                evaluation_cache.hidden_or_chance_partitioned_keys.add(cache_key)
        return (
            self._apply_cached_leaf(
                node,
                cached,
                combos=combos,
                factorized=factorized,
                action_prefix=prefix,
                total_action_count=total_action_count,
            ),
            True,
        )

    def _evaluate_node(
        self,
        node: BeliefNode,
        obs: Any,
        *,
        root_seat: int,
        particle: Optional[HiddenStateParticle],
        branch: _BranchHistory,
        action_prefix: Sequence[int] = (),
    ) -> float:
        """Evaluate a standalone node without cross-node cache reuse."""

        value, _issued_model_forward = self._evaluate_node_cached(
            node,
            obs,
            root_seat=root_seat,
            particle=particle,
            branch=branch,
            action_prefix=action_prefix,
        )
        return value

    @staticmethod
    def _factorized_action_complete(
        obs: Any,
        prefix: Sequence[int],
        candidate: Sequence[int],
    ) -> bool:
        prefix_list = list(prefix)
        candidate_list = list(candidate)
        if candidate_list == prefix_list:
            return True
        next_candidates = features.factorized_action_candidates(
            _raw_observation(obs), candidate_list
        )
        return len(next_candidates) == 1 and next_candidates[0] == candidate_list

    @staticmethod
    def _factorized_key(fingerprint: str, prefix: Sequence[int]) -> str:
        encoded = ",".join(str(index) for index in prefix)
        return f"{fingerprint}|action-prefix:{encoded}"

    def _select_edge(self, node: BeliefNode, root_seat: int) -> BeliefEdge:
        if not node.edges:
            raise ValueError("cannot select from an unexpanded information node")
        sign = 1.0 if node.actor == root_seat else -1.0
        scale = self.puct_c * math.sqrt(max(1, node.visit))
        return max(
            node.edges,
            key=lambda edge: (
                sign * edge.q() + scale * edge.prior / (1 + edge.visit),
                edge.prior,
                tuple(-item for item in edge.action),
            ),
        )

    @staticmethod
    def _backup(
        nodes: Sequence[BeliefNode],
        edges: Sequence[BeliefEdge],
        value: float,
    ) -> None:
        for node in nodes:
            node.visit += 1
            node.total += value
        for edge in edges:
            edge.visit += 1
            edge.total += value

    @staticmethod
    def _fork_branch_history(branch: _BranchHistory) -> _BranchHistory:
        """Copy only the mutable branch-local history containers.

        Exact finite chance evaluates every forced successor.  Those child
        evaluations must not append model-history tokens into one another's
        branch merely because they originate at the same chance parent.
        """

        return _BranchHistory(
            boards={actor: list(rows) for actor, rows in branch.boards.items()},
            previous_actions={
                actor: list(rows)
                for actor, rows in branch.previous_actions.items()
            },
            last_action=dict(branch.last_action),
        )

    @staticmethod
    def _chance_successor_observation(successor: Any) -> Any:
        """Return the public observation bound to an opaque chance child."""

        if hasattr(successor, "observation"):
            return successor.observation
        if isinstance(successor, dict) and "observation" in successor:
            return successor["observation"]
        raise ValueError("exact finite-chance successor has no observation")

    @staticmethod
    def _receipt_is_present(receipt: Any) -> bool:
        return isinstance(receipt, str) and bool(receipt.strip())

    def _validate_exact_finite_chance_expansion(
        self,
        expansion: Any,
    ) -> ExactFiniteChanceExpansion:
        """Validate the narrow r219 exactness preconditions, or fail closed."""

        if not isinstance(expansion, ExactFiniteChanceExpansion):
            raise ValueError("capability did not return ExactFiniteChanceExpansion")
        if not self._receipt_is_present(expansion.force_enumeration_receipt):
            raise ValueError("missing force-enumeration receipt")
        if not self._receipt_is_present(expansion.probability_receipt):
            raise ValueError("missing probability receipt")
        outcomes = tuple(expansion.outcomes)
        if not 2 <= len(outcomes) <= 6:
            raise ValueError("outcome count must be between two and six")
        labels: set[str] = set()
        successor_receipts: set[str] = set()
        successor_objects: set[int] = set()
        probability_sum = Fraction()
        for outcome in outcomes:
            if not isinstance(outcome, ExactFiniteChanceOutcome):
                raise ValueError("chance outcomes must use ExactFiniteChanceOutcome")
            if not isinstance(outcome.label, str) or not outcome.label:
                raise ValueError("chance outcome label is missing")
            if outcome.label in labels:
                raise ValueError("chance outcome labels are not unique")
            labels.add(outcome.label)
            if not isinstance(outcome.probability, Fraction) or outcome.probability <= 0:
                raise ValueError("chance probabilities must be positive Fractions")
            probability_sum += outcome.probability
            if outcome.successor is None:
                raise ValueError("chance outcome has no independently advanceable child")
            if not self._receipt_is_present(outcome.successor_receipt):
                raise ValueError("chance outcome has no successor receipt")
            if outcome.successor_receipt in successor_receipts:
                raise ValueError("chance successor receipts are not unique")
            successor_receipts.add(outcome.successor_receipt)
            if id(outcome.successor) in successor_objects:
                raise ValueError("chance outcomes reused one successor object")
            successor_objects.add(id(outcome.successor))
            if not self._receipt_is_present(outcome.future_legality_receipt):
                raise ValueError("chance outcome has no future-legality receipt")
            child_observation = self._chance_successor_observation(
                outcome.successor
            )
            _raw_observation(child_observation)
            # A second unresolved chance node needs its own complete forced
            # distribution.  This small adapter deliberately does not recurse
            # through an opaque chain and must not pretend a leaf evaluation is
            # an exact continuation of it.
            if (
                self._terminal_value(child_observation, 0) is None
                and is_explicit_chance(child_observation)
            ):
                raise ValueError("nested unresolved chance child")
            if (
                self._terminal_value(child_observation, 0) is None
                and not self._can_advance_exact_finite_chance_successor(
                    outcome.successor
                )
            ):
                raise ValueError(
                    "nonterminal chance child is not independently advanceable"
                )
        if probability_sum != Fraction(1, 1):
            raise ValueError("chance probabilities do not sum exactly to one")
        return expansion

    def _try_exact_finite_chance_expansion(
        self,
        search_state: Any,
    ) -> tuple[Optional[ExactFiniteChanceExpansion], Optional[str]]:
        """Ask a capability for a complete finite distribution, never guess.

        The public libcg wrapper presently has no such facility.  A missing,
        unavailable, or malformed optional binding therefore returns an
        explicit reason and makes the caller stop at a labelled leaf boundary.
        """

        capability = self._exact_finite_chance_capability()
        if capability is None:
            return None, "no_exact_finite_chance_capability"
        enumerate_outcomes = getattr(
            capability, "enumerate_exact_finite_chance", None
        )
        if not callable(enumerate_outcomes):
            return None, "capability_has_no_force_enumerator"
        try:
            expansion = enumerate_outcomes(search_state)
        except ExactFiniteChanceUnavailable as exc:
            detail = str(exc).strip() or "capability_reported_unforceable"
            return None, f"capability_unforceable:{detail}"
        if expansion is None:
            return None, "capability_reported_unforceable"
        try:
            return self._validate_exact_finite_chance_expansion(expansion), None
        except (TypeError, ValueError) as exc:
            return None, f"invalid_exact_finite_chance_capability:{exc}"

    def _exact_finite_chance_capability(self) -> Any:
        """Return the one optional capability without making it mandatory."""

        capability = getattr(self, "exact_finite_chance_capability", None)
        if capability is None:
            capability = getattr(self, "finite_chance_capability", None)
        return capability

    def _can_advance_exact_finite_chance_successor(self, successor: Any) -> bool:
        """Require a real post-child action route for nonterminal outcomes."""

        capability = self._exact_finite_chance_capability()
        if callable(
            getattr(capability, "advance_exact_finite_chance_successor", None)
        ):
            return True
        if isinstance(successor, dict):
            return successor.get("searchId") is not None
        return getattr(successor, "searchId", None) is not None

    def _advance_exact_finite_chance_successor(
        self,
        successor: Any,
        action: Sequence[int],
    ) -> Any:
        """Advance a sealed child through its own safe backend capability."""

        capability = self._exact_finite_chance_capability()
        advance = getattr(
            capability, "advance_exact_finite_chance_successor", None
        )
        if callable(advance):
            return advance(successor, list(action))
        search_id = (
            successor.get("searchId")
            if isinstance(successor, dict)
            else getattr(successor, "searchId", None)
        )
        if search_id is None:
            raise ExactFiniteChanceUnavailable(
                "exact chance child has no independently advanceable route"
            )
        backend = getattr(self, "search_backend", cg_env)
        return backend.search_step(search_id, list(action))

    def _evaluate_unforceable_chance_boundary(
        self,
        observation: Any,
        *,
        root_seat: int,
        particle: HiddenStateParticle,
        branch: _BranchHistory,
        depth: int,
        evaluation_cache: _DecisionEvaluationCache,
        scenario_prefix: str,
        reason: str,
    ) -> tuple[float, bool]:
        """Leaf-evaluate an unresolved chance menu without selecting it.

        A missing/incomplete forcing capability means the distribution is not
        known.  The truthful continuation is a frozen neural leaf at the
        pre-random state.  This temporary node is intentionally not linked to
        the tree: retaining its coin/die menu as PUCT edges would permit a
        later guessed private outcome.
        """

        boundary_branch = self._fork_branch_history(branch)
        self._record_observation(
            observation,
            root_seat=root_seat,
            particle=particle,
            branch=boundary_branch,
        )
        boundary = BeliefNode(
            fingerprint=information_state_fingerprint(observation),
            actor=self._actor(observation),
            depth=depth,
        )
        return self._evaluate_node_cached(
            boundary,
            observation,
            root_seat=root_seat,
            particle=particle,
            branch=boundary_branch,
            evaluation_cache=evaluation_cache,
            hidden_or_chance_scenario="|".join(
                (
                    scenario_prefix,
                    "unforceable-chance-boundary:" + reason,
                )
            ),
        )

    @staticmethod
    def _exact_chance_outcome_key(
        *,
        parent_fingerprint: str,
        expansion: ExactFiniteChanceExpansion,
        outcome: ExactFiniteChanceOutcome,
        child_fingerprint: str,
    ) -> str:
        """Keep exact chance children distinct inside a public-history edge."""

        encoded = json.dumps(
            {
                "schema": "poke_bot.belief_mcts.exact_finite_chance_child/v1",
                "parent": parent_fingerprint,
                "force_enumeration_receipt": expansion.force_enumeration_receipt,
                "probability_receipt": expansion.probability_receipt,
                "outcome_label": outcome.label,
                "successor_receipt": outcome.successor_receipt,
                "future_legality_receipt": outcome.future_legality_receipt,
                "child_information_state": child_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return "exact-chance:" + hashlib.sha256(encoded).hexdigest()

    def _evaluate_exact_finite_chance(
        self,
        *,
        parent: BeliefNode,
        edge: BeliefEdge,
        expansion: ExactFiniteChanceExpansion,
        root_seat: int,
        particle: HiddenStateParticle,
        branch: _BranchHistory,
        depth: int,
        evaluation_cache: _DecisionEvaluationCache,
        scenario_prefix: str,
    ) -> _FiniteChanceRollout:
        """Evaluate every sealed child and continue a child search rollout.

        The first visit to a forced child evaluates its legal decision leaf.
        Later visits select and advance that child's own PUCT edge, exactly as
        a normal MCTS simulation would.  Thus a coin/die node is not a frozen
        one-ply reranker: each nonzero-probability outcome keeps growing its
        independent subtree while the parent receives their exact weighted
        expectation.
        """

        weighted_values: list[float] = []
        terminal_results_seen = 0
        leaf_evaluations = 0
        simulator_transitions = len(expansion.outcomes)
        max_child_simulator_steps = 0
        finite_chance_enumerations = 1
        finite_chance_outcomes_enumerated = len(expansion.outcomes)
        finite_chance_weighted_backup_count = 1
        forced_successor_transitions = len(expansion.outcomes)
        chance_samples = 0
        sampled_unforceable_chance_nodes = 0
        sampled_unforceable_chance_reasons: dict[str, int] = {}
        unforceable_chance_boundary_nodes = 0
        unforceable_chance_boundary_leaf_evaluations = 0
        unforceable_chance_boundary_reasons: dict[str, int] = {}
        for outcome in expansion.outcomes:
            child_rollout = self._rollout_exact_finite_chance_child(
                parent=parent,
                parent_edge=edge,
                expansion=expansion,
                outcome=outcome,
                root_seat=root_seat,
                particle=particle,
                branch=branch,
                depth=depth,
                evaluation_cache=evaluation_cache,
                scenario_prefix=scenario_prefix,
            )
            weighted_values.append(
                float(outcome.probability) * child_rollout.value
            )
            terminal_results_seen += child_rollout.terminal_results_seen
            leaf_evaluations += child_rollout.leaf_evaluations
            simulator_transitions += child_rollout.simulator_transitions
            max_child_simulator_steps = max(
                max_child_simulator_steps,
                child_rollout.max_simulator_steps,
            )
            finite_chance_enumerations += child_rollout.finite_chance_enumerations
            finite_chance_outcomes_enumerated += (
                child_rollout.finite_chance_outcomes_enumerated
            )
            finite_chance_weighted_backup_count += (
                child_rollout.finite_chance_weighted_backup_count
            )
            forced_successor_transitions += (
                child_rollout.finite_chance_forced_successor_transitions
            )
            chance_samples += child_rollout.chance_samples
            sampled_unforceable_chance_nodes += (
                child_rollout.sampled_unforceable_chance_nodes
            )
            for reason, count in (
                child_rollout.sampled_unforceable_chance_reasons.items()
            ):
                sampled_unforceable_chance_reasons[reason] = (
                    sampled_unforceable_chance_reasons.get(reason, 0) + count
                )
            unforceable_chance_boundary_nodes += (
                child_rollout.unforceable_chance_boundary_nodes
            )
            unforceable_chance_boundary_leaf_evaluations += (
                child_rollout.unforceable_chance_boundary_leaf_evaluations
            )
            for reason, count in (
                child_rollout.unforceable_chance_boundary_reasons.items()
            ):
                unforceable_chance_boundary_reasons[reason] = (
                    unforceable_chance_boundary_reasons.get(reason, 0) + count
                )
        return _FiniteChanceRollout(
            value=math.fsum(weighted_values),
            terminal_results_seen=terminal_results_seen,
            leaf_evaluations=leaf_evaluations,
            simulator_transitions=simulator_transitions,
            max_simulator_steps=1 + max_child_simulator_steps,
            finite_chance_enumerations=finite_chance_enumerations,
            finite_chance_outcomes_enumerated=finite_chance_outcomes_enumerated,
            finite_chance_weighted_backup_count=finite_chance_weighted_backup_count,
            finite_chance_forced_successor_transitions=(
                forced_successor_transitions
            ),
            unforceable_chance_boundary_nodes=(
                unforceable_chance_boundary_nodes
            ),
            unforceable_chance_boundary_leaf_evaluations=(
                unforceable_chance_boundary_leaf_evaluations
            ),
            unforceable_chance_boundary_reasons=(
                unforceable_chance_boundary_reasons
            ),
            chance_samples=chance_samples,
            sampled_unforceable_chance_nodes=(
                sampled_unforceable_chance_nodes
            ),
            sampled_unforceable_chance_reasons=(
                sampled_unforceable_chance_reasons
            ),
        )

    def _rollout_exact_finite_chance_child(
        self,
        *,
        parent: BeliefNode,
        parent_edge: BeliefEdge,
        expansion: ExactFiniteChanceExpansion,
        outcome: ExactFiniteChanceOutcome,
        root_seat: int,
        particle: HiddenStateParticle,
        branch: _BranchHistory,
        depth: int,
        evaluation_cache: _DecisionEvaluationCache,
        scenario_prefix: str,
    ) -> _FiniteChanceRollout:
        """Run one normal MCTS descent from one independently forced child."""

        child_state = outcome.successor
        child_observation = self._chance_successor_observation(child_state)
        terminal = self._terminal_value(child_observation, root_seat)
        if terminal is not None:
            return _FiniteChanceRollout(
                value=float(terminal),
                terminal_results_seen=1,
            )

        child_branch = self._fork_branch_history(branch)
        self._record_observation(
            child_observation,
            root_seat=root_seat,
            particle=particle,
            branch=child_branch,
        )
        child_fingerprint = information_state_fingerprint(child_observation)
        child_key = self._exact_chance_outcome_key(
            parent_fingerprint=parent.fingerprint,
            expansion=expansion,
            outcome=outcome,
            child_fingerprint=child_fingerprint,
        )
        child = parent_edge.outcomes.get(child_key)
        if child is None:
            child = BeliefNode(
                fingerprint=child_fingerprint,
                actor=self._actor(child_observation),
                depth=depth,
            )
            parent_edge.outcomes[child_key] = child
        scenario_parts = [
            scenario_prefix,
            "exact-chance-force:" + expansion.force_enumeration_receipt,
            "exact-chance-probability:" + expansion.probability_receipt,
            "exact-chance-outcome:" + outcome.label,
            "exact-chance-successor:" + outcome.successor_receipt,
            "exact-chance-legality:" + outcome.future_legality_receipt,
        ]
        if not child.network_evaluated:
            value, forward_issued = self._evaluate_node_cached(
                child,
                child_observation,
                root_seat=root_seat,
                particle=particle,
                branch=child_branch,
                evaluation_cache=evaluation_cache,
                hidden_or_chance_scenario="|".join(scenario_parts),
            )
            self._backup([child], [], value)
            return _FiniteChanceRollout(
                value=float(value),
                leaf_evaluations=int(forward_issued),
            )

        current = child
        search_state = child_state
        nodes = [child]
        edges: list[BeliefEdge] = []
        value: Optional[float] = None
        terminal_results_seen = 0
        leaf_evaluations = 0
        simulator_transitions = 0
        simulator_steps = 0
        finite_chance_enumerations = 0
        finite_chance_outcomes_enumerated = 0
        finite_chance_weighted_backup_count = 0
        forced_successor_transitions = 0
        chance_samples = 0
        sampled_unforceable_chance_nodes = 0
        sampled_unforceable_chance_reasons: dict[str, int] = {}
        unforceable_chance_boundary_nodes = 0
        unforceable_chance_boundary_leaf_evaluations = 0
        unforceable_chance_boundary_reasons: dict[str, int] = {}
        for descendant_depth in range(depth + 1, self.max_depth + 1):
            selected_edge = self._select_edge(current, root_seat)
            edges.append(selected_edge)
            if current.factorized and not self._factorized_action_complete(
                search_state.observation,
                current.action_prefix,
                selected_edge.action,
            ):
                fingerprint = self._factorized_key(
                    information_state_fingerprint(search_state.observation),
                    selected_edge.action,
                )
                prefix_child = selected_edge.outcomes.get(fingerprint)
                if prefix_child is None:
                    prefix_child = BeliefNode(
                        fingerprint=fingerprint,
                        actor=current.actor,
                        depth=descendant_depth,
                    )
                    selected_edge.outcomes[fingerprint] = prefix_child
                nodes.append(prefix_child)
                current = prefix_child
                if not prefix_child.network_evaluated:
                    value, forward_issued = self._evaluate_node_cached(
                        prefix_child,
                        search_state.observation,
                        root_seat=root_seat,
                        particle=particle,
                        branch=child_branch,
                        action_prefix=selected_edge.action,
                        evaluation_cache=evaluation_cache,
                        hidden_or_chance_scenario="|".join(scenario_parts),
                    )
                    leaf_evaluations += int(forward_issued)
                    break
                continue
            child_branch.last_action[current.actor] = features.build_option_tokens(
                _raw_observation(search_state.observation), [selected_edge.action]
            )
            search_state = self._advance_exact_finite_chance_successor(
                search_state,
                selected_edge.action,
            )
            simulator_transitions += 1
            simulator_steps += 1
            scenario_parts.append(
                "action:" + ",".join(str(index) for index in selected_edge.action)
            )
            if is_explicit_chance(search_state.observation):
                nested_expansion, unforceable_reason = (
                    self._try_exact_finite_chance_expansion(search_state)
                )
                if nested_expansion is not None:
                    nested = self._evaluate_exact_finite_chance(
                        parent=current,
                        edge=selected_edge,
                        expansion=nested_expansion,
                        root_seat=root_seat,
                        particle=particle,
                        branch=child_branch,
                        depth=descendant_depth + 1,
                        evaluation_cache=evaluation_cache,
                        scenario_prefix="|".join(scenario_parts),
                    )
                    value = nested.value
                    terminal_results_seen += nested.terminal_results_seen
                    leaf_evaluations += nested.leaf_evaluations
                    simulator_transitions += nested.simulator_transitions
                    simulator_steps += nested.max_simulator_steps
                    finite_chance_enumerations += nested.finite_chance_enumerations
                    finite_chance_outcomes_enumerated += (
                        nested.finite_chance_outcomes_enumerated
                    )
                    finite_chance_weighted_backup_count += (
                        nested.finite_chance_weighted_backup_count
                    )
                    forced_successor_transitions += (
                        nested.finite_chance_forced_successor_transitions
                    )
                    chance_samples += nested.chance_samples
                    sampled_unforceable_chance_nodes += (
                        nested.sampled_unforceable_chance_nodes
                    )
                    for reason, count in (
                        nested.sampled_unforceable_chance_reasons.items()
                    ):
                        sampled_unforceable_chance_reasons[reason] = (
                            sampled_unforceable_chance_reasons.get(reason, 0)
                            + count
                        )
                    unforceable_chance_boundary_nodes += (
                        nested.unforceable_chance_boundary_nodes
                    )
                    unforceable_chance_boundary_leaf_evaluations += (
                        nested.unforceable_chance_boundary_leaf_evaluations
                    )
                    for reason, count in (
                        nested.unforceable_chance_boundary_reasons.items()
                    ):
                        unforceable_chance_boundary_reasons[reason] = (
                            unforceable_chance_boundary_reasons.get(reason, 0)
                            + count
                        )
                    break
                # Do not advance a manually selectable coin/die node.  The
                # private simulator has not attested every outcome, exact
                # probability, independent child, and future legal set, so
                # the chance menu itself is the truthful neural leaf.
                reason = (
                    unforceable_reason
                    or "unforceable_exact_finite_chance"
                )
                selected_edge.saw_sampled_or_opaque_chance = True
                value, forward_issued = self._evaluate_unforceable_chance_boundary(
                    search_state.observation,
                    root_seat=root_seat,
                    particle=particle,
                    branch=child_branch,
                    depth=descendant_depth + 1,
                    evaluation_cache=evaluation_cache,
                    scenario_prefix="|".join(scenario_parts),
                    reason=reason,
                )
                leaf_evaluations += int(forward_issued)
                unforceable_chance_boundary_nodes += 1
                unforceable_chance_boundary_leaf_evaluations += int(
                    forward_issued
                )
                unforceable_chance_boundary_reasons[reason] = (
                    unforceable_chance_boundary_reasons.get(reason, 0) + 1
                )
                break
            terminal = self._terminal_value(search_state.observation, root_seat)
            if terminal is not None:
                value = float(terminal)
                terminal_results_seen += 1
                break
            self._record_observation(
                search_state.observation,
                root_seat=root_seat,
                particle=particle,
                branch=child_branch,
            )
            fingerprint = information_state_fingerprint(search_state.observation)
            successor = selected_edge.outcomes.get(fingerprint)
            if successor is None:
                successor = BeliefNode(
                    fingerprint=fingerprint,
                    actor=self._actor(search_state.observation),
                    depth=descendant_depth,
                )
                selected_edge.outcomes[fingerprint] = successor
            nodes.append(successor)
            current = successor
            if not successor.network_evaluated:
                value, forward_issued = self._evaluate_node_cached(
                    successor,
                    search_state.observation,
                    root_seat=root_seat,
                    particle=particle,
                    branch=child_branch,
                    evaluation_cache=evaluation_cache,
                    hidden_or_chance_scenario="|".join(scenario_parts),
                )
                leaf_evaluations += int(forward_issued)
                break
        if value is None:
            # Reaching the emergency depth ceiling is not a reason to fabricate
            # a child value.  A previously frozen bootstrap is a valid leaf;
            # without one, this particular exact child cannot be completed.
            if current.bootstrap_value is None:
                raise RuntimeError(
                    "exact finite-chance child reached depth ceiling without "
                    "a backed-up leaf value"
                )
            value = current.bootstrap_value
        self._backup(nodes, edges, value)
        return _FiniteChanceRollout(
            value=float(value),
            terminal_results_seen=terminal_results_seen,
            leaf_evaluations=leaf_evaluations,
            simulator_transitions=simulator_transitions,
            max_simulator_steps=simulator_steps,
            finite_chance_enumerations=finite_chance_enumerations,
            finite_chance_outcomes_enumerated=finite_chance_outcomes_enumerated,
            finite_chance_weighted_backup_count=finite_chance_weighted_backup_count,
            finite_chance_forced_successor_transitions=(
                forced_successor_transitions
            ),
            unforceable_chance_boundary_nodes=(
                unforceable_chance_boundary_nodes
            ),
            unforceable_chance_boundary_leaf_evaluations=(
                unforceable_chance_boundary_leaf_evaluations
            ),
            unforceable_chance_boundary_reasons=(
                unforceable_chance_boundary_reasons
            ),
            chance_samples=chance_samples,
            sampled_unforceable_chance_nodes=(
                sampled_unforceable_chance_nodes
            ),
            sampled_unforceable_chance_reasons=(
                sampled_unforceable_chance_reasons
            ),
        )

    def _sample_chance_until_decision(
        self,
        search_state,
    ) -> tuple[Any, int, tuple[str, ...]]:
        """Reject obsolete private-chance sampling rather than performing it.

        Search paths must reach an unresolved manual coin/die only through
        :meth:`_evaluate_unforceable_chance_boundary`.  Keeping this guard
        protects the invariant against an accidental future call site: neither
        RNG nor ``search_step`` may be used to fabricate an outcome.
        """

        del search_state
        raise RuntimeError(
            "private chance sampling is prohibited; use the pre-random leaf "
            "boundary or a sealed exact finite-chance capability"
        )

    def search(
        self,
        obs_dict: dict[str, Any],
        *,
        belief_history: PublicBeliefHistory,
        root_history_boards: Sequence[features.SparseVector],
        root_history_previous_actions: Sequence[
            Optional[features.SparseVector]
        ],
        clock: Optional[GameClock] = None,
        max_sims: int = 128,
        move_time_s: float = 8.0,
        temperature: float = 1.0,
        root_neural_priors: Optional[NeuralBeliefPriors] = None,
    ) -> MCTSResult:
        """Run one deadline-bounded trusted search with adaptive sim cap."""
        requested_sims = int(max_sims)
        if requested_sims < self.min_trusted_sims:
            raise ValueError(
                f"trusted belief MCTS requires max_sims>={self.min_trusted_sims}"
            )
        configured_budget = max(0.05, float(move_time_s))
        move_budget = configured_budget
        if clock is not None:
            move_budget = min(
                configured_budget,
                clock.next_move_budget(configured_budget),
            )
        # Keep the caller's explicit trust floor, but shed optional ramped
        # simulations when the per-game clock grants less than the configured
        # move slice.
        ratio = min(1.0, move_budget / configured_budget)
        adaptive_sims = max(
            self.min_trusted_sims,
            min(requested_sims, int(math.ceil(requested_sims * ratio))),
        )
        set_deadline = getattr(self.leaf_eval, "set_deadline", None)
        if callable(set_deadline):
            set_deadline(time.monotonic() + move_budget)
        try:
            return self._search_impl(
                obs_dict,
                belief_history=belief_history,
                root_history_boards=root_history_boards,
                root_history_previous_actions=root_history_previous_actions,
                clock=clock,
                max_sims=adaptive_sims,
                requested_max_sims=requested_sims,
                move_time_s=move_budget,
                temperature=temperature,
                root_neural_priors=root_neural_priors,
            )
        finally:
            if callable(set_deadline):
                set_deadline(None)

    def _search_impl(
        self,
        obs_dict: dict[str, Any],
        *,
        belief_history: PublicBeliefHistory,
        root_history_boards: Sequence[features.SparseVector],
        root_history_previous_actions: Sequence[
            Optional[features.SparseVector]
        ],
        clock: Optional[GameClock] = None,
        max_sims: int = 128,
        requested_max_sims: Optional[int] = None,
        move_time_s: float = 8.0,
        temperature: float = 1.0,
        root_neural_priors: Optional[NeuralBeliefPriors] = None,
    ) -> MCTSResult:
        assert_deployment_observation(obs_dict)
        features.assert_info_set(obs_dict)
        obs = cg_env.to_observation(obs_dict)
        if obs.current is None:
            raise ValueError("belief MCTS requires post-setup observation")
        root_seat = int(obs.current.yourIndex)
        if len(root_history_boards) != len(root_history_previous_actions):
            raise ValueError("root history/action lengths differ")
        belief_history.observe(obs_dict)
        move_budget = float(move_time_s)
        if clock is not None:
            move_budget = min(move_budget, max(0.05, clock.remaining_s))
        sims_plan = int(max_sims)
        if sims_plan < self.min_trusted_sims:
            raise ValueError(
                f"trusted belief MCTS requires max_sims>={self.min_trusted_sims}"
            )
        search_backend = getattr(self, "search_backend", cg_env)
        stop_requested = getattr(self, "stop_requested", lambda: False)

        root = BeliefNode(
            fingerprint=information_state_fingerprint(obs_dict),
            actor=root_seat,
            depth=0,
        )
        root_branch = _BranchHistory(
            boards={
                root_seat: list(root_history_boards),
                1 - root_seat: [],
            },
            previous_actions={
                root_seat: list(root_history_previous_actions),
                1 - root_seat: [],
            },
            last_action={root_seat: None, 1 - root_seat: None},
        )
        telemetry_marker = self._telemetry_mark()
        started = time.perf_counter()
        evaluation_cache = _DecisionEvaluationCache()
        _root_value, root_forward_issued = self._evaluate_node_cached(
            root,
            obs_dict,
            root_seat=root_seat,
            particle=None,
            branch=root_branch,
            evaluation_cache=evaluation_cache,
            # Root is evaluated once before any private simulator world exists.
            # It is still isolated to this search call by ``evaluation_cache``.
            hidden_or_chance_scenario="public-root",
        )
        leaf_evaluations = int(root_forward_issued)
        chance_samples = 0
        finite_chance_enumerations = 0
        finite_chance_outcomes_enumerated = 0
        finite_chance_weighted_backup_count = 0
        finite_chance_forced_successor_transitions = 0
        unforceable_chance_boundary_nodes = 0
        unforceable_chance_boundary_leaf_evaluations = 0
        unforceable_chance_boundary_reasons: dict[str, int] = {}
        sampled_unforceable_chance_nodes = 0
        sampled_unforceable_chance_reasons: dict[str, int] = {}
        simulator_transitions = 0
        deterministic_successor_expansions = 0
        exact_terminal_results_seen = 0
        value_backups = 0
        max_simulator_search_depth = 0
        multi_step_simulations = 0
        sims_run = 0
        convergence_best_action: tuple[int, ...] | None = None
        convergence_stable_backups = 0
        root_action_stable = False
        root_stability_receipt: dict[str, Any] | None = None
        particle_decks: set[str] = set()
        particle_states: set[str] = set()
        particle_attempts = 0
        # Root-only NN reweight of the empirical posterior / hand fill.
        neural_priors = root_neural_priors or self._root_neural_priors(
            root_history_boards=root_history_boards,
            root_history_previous_actions=root_history_previous_actions,
        )
        particle_bank = [
            self.posterior.sample_particle(
                obs_dict,
                own_deck=self.own_deck,
                history=belief_history,
                rng=self.rng,
                neural=neural_priors,
            )
            for _ in range(min(self.particle_count, sims_plan))
        ]
        self.rng.shuffle(particle_bank)
        particle_support_modes = sorted(
            {particle.support_mode for particle in particle_bank}
        )
        particle_support_repairs = sum(
            int(particle.support_repairs) for particle in particle_bank
        )
        while (
            sims_run < sims_plan
            and time.perf_counter() - started < move_budget
            and not stop_requested()
        ):
            particle = particle_bank[particle_attempts % len(particle_bank)]
            particle_attempts += 1
            particle_decks.add(particle.opponent_deck_digest)
            particle_states.add(
                hashlib.sha256(
                    json.dumps(
                        particle.search_inputs,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            search_state = search_backend.search_begin(
                obs_dict, particle.search_inputs, manual_coin=True
            )
            # The public libcg binding currently offers no attested complete
            # semantic-state/RNG identity.  Including the fresh private search
            # world in the scenario keeps this cache from merging independent
            # worlds that merely look alike publicly.  Node-local expansion
            # still prevents a repeat forward for an already evaluated node.
            scenario_parts = [
                _hidden_particle_identity(particle),
                f"search-world:{search_state.searchId}",
            ]
            branch = _BranchHistory(
                boards={
                    root_seat: list(root_history_boards),
                    1 - root_seat: [],
                },
                previous_actions={
                    root_seat: list(root_history_previous_actions),
                    1 - root_seat: [],
                },
                last_action={root_seat: None, 1 - root_seat: None},
            )
            matchup_shadow_branch = (
                self.matchup_shadow_router.fork()
                if self.matchup_shadow_router is not None
                else None
            )
            nodes = [root]
            edges: list[BeliefEdge] = []
            current = root
            value: Optional[float] = None
            simulator_steps = 0
            try:
                for depth in range(1, self.max_depth + 1):
                    if (
                        time.perf_counter() - started >= move_budget
                        or stop_requested()
                    ):
                        break
                    edge = self._select_edge(current, root_seat)
                    edges.append(edge)
                    if current.factorized and not self._factorized_action_complete(
                        search_state.observation,
                        current.action_prefix,
                        edge.action,
                    ):
                        fingerprint = self._factorized_key(
                            information_state_fingerprint(
                                search_state.observation
                            ),
                            edge.action,
                        )
                        child = edge.outcomes.get(fingerprint)
                        if child is None:
                            child = BeliefNode(
                                fingerprint=fingerprint,
                                actor=current.actor,
                                depth=depth,
                            )
                            edge.outcomes[fingerprint] = child
                        nodes.append(child)
                        current = child
                        if not child.network_evaluated:
                            value, forward_issued = self._evaluate_node_cached(
                                child,
                                search_state.observation,
                                root_seat=root_seat,
                                particle=particle,
                                branch=branch,
                                action_prefix=edge.action,
                                evaluation_cache=evaluation_cache,
                                hidden_or_chance_scenario="|".join(
                                    scenario_parts
                                ),
                            )
                            leaf_evaluations += int(forward_issued)
                            break
                        continue
                    branch.last_action[current.actor] = features.build_option_tokens(
                        _raw_observation(search_state.observation), [edge.action]
                    )
                    search_state = search_backend.search_step(
                        search_state.searchId, edge.action
                    )
                    simulator_transitions += 1
                    simulator_steps += 1
                    scenario_parts.append(
                        "action:" + ",".join(str(index) for index in edge.action)
                    )
                    if is_explicit_chance(search_state.observation):
                        exact_expansion, unforceable_reason = (
                            self._try_exact_finite_chance_expansion(search_state)
                        )
                        if exact_expansion is not None:
                            # One parent action now has one complete expected
                            # backup.  Every nonzero-probability child was
                            # evaluated first; none is dropped by a heuristic
                            # threshold or replaced with an RNG sample.
                            exact_rollout = self._evaluate_exact_finite_chance(
                                parent=current,
                                edge=edge,
                                expansion=exact_expansion,
                                root_seat=root_seat,
                                particle=particle,
                                branch=branch,
                                depth=depth + 1,
                                evaluation_cache=evaluation_cache,
                                scenario_prefix="|".join(scenario_parts),
                            )
                            finite_chance_enumerations += (
                                exact_rollout.finite_chance_enumerations
                            )
                            finite_chance_outcomes_enumerated += (
                                exact_rollout.finite_chance_outcomes_enumerated
                            )
                            finite_chance_weighted_backup_count += (
                                exact_rollout.finite_chance_weighted_backup_count
                            )
                            finite_chance_forced_successor_transitions += (
                                exact_rollout.finite_chance_forced_successor_transitions
                            )
                            leaf_evaluations += exact_rollout.leaf_evaluations
                            exact_terminal_results_seen += (
                                exact_rollout.terminal_results_seen
                            )
                            chance_samples += exact_rollout.chance_samples
                            sampled_unforceable_chance_nodes += (
                                exact_rollout.sampled_unforceable_chance_nodes
                            )
                            for reason, count in (
                                exact_rollout.sampled_unforceable_chance_reasons.items()
                            ):
                                sampled_unforceable_chance_reasons[reason] = (
                                    sampled_unforceable_chance_reasons.get(reason, 0)
                                    + count
                                )
                            unforceable_chance_boundary_nodes += (
                                exact_rollout.unforceable_chance_boundary_nodes
                            )
                            unforceable_chance_boundary_leaf_evaluations += (
                                exact_rollout.unforceable_chance_boundary_leaf_evaluations
                            )
                            for reason, count in (
                                exact_rollout.unforceable_chance_boundary_reasons.items()
                            ):
                                unforceable_chance_boundary_reasons[reason] = (
                                    unforceable_chance_boundary_reasons.get(reason, 0)
                                    + count
                                )
                            simulator_transitions += (
                                exact_rollout.simulator_transitions
                            )
                            simulator_steps += exact_rollout.max_simulator_steps
                            value = exact_rollout.value
                            break
                        # No sealed all-outcome distribution is available. Do
                        # not hunt for a seed, select heads/tails, or advance a
                        # guessed outcome. The chance menu itself is the frozen
                        # pre-random leaf; reality may re-root a later search.
                        reason = (
                            unforceable_reason
                            or "unforceable_exact_finite_chance"
                        )
                        edge.saw_sampled_or_opaque_chance = True
                        value, forward_issued = (
                            self._evaluate_unforceable_chance_boundary(
                                search_state.observation,
                                root_seat=root_seat,
                                particle=particle,
                                branch=branch,
                                depth=depth + 1,
                                evaluation_cache=evaluation_cache,
                                scenario_prefix="|".join(scenario_parts),
                                reason=reason,
                            )
                        )
                        leaf_evaluations += int(forward_issued)
                        unforceable_chance_boundary_nodes += 1
                        unforceable_chance_boundary_leaf_evaluations += int(
                            forward_issued
                        )
                        unforceable_chance_boundary_reasons[reason] = (
                            unforceable_chance_boundary_reasons.get(reason, 0) + 1
                        )
                        break
                    terminal = self._terminal_value(
                        search_state.observation, root_seat
                    )
                    if terminal is not None:
                        exact_terminal_results_seen += 1
                        value = terminal
                        break
                    if (
                        matchup_shadow_branch is not None
                        and self._actor(search_state.observation) == root_seat
                    ):
                        matchup_shadow_branch.observe(
                            search_state.observation,
                            scope="belief_search_branch",
                            depth=depth,
                        )
                    self._record_observation(
                        search_state.observation,
                        root_seat=root_seat,
                        particle=particle,
                        branch=branch,
                    )
                    fingerprint = information_state_fingerprint(
                        search_state.observation
                    )
                    child = edge.outcomes.get(fingerprint)
                    if child is None:
                        child = BeliefNode(
                            fingerprint=fingerprint,
                            actor=self._actor(search_state.observation),
                            depth=depth,
                        )
                        edge.outcomes[fingerprint] = child
                        deterministic_successor_expansions += 1
                    nodes.append(child)
                    current = child
                    if not child.network_evaluated:
                        value, forward_issued = self._evaluate_node_cached(
                            child,
                            search_state.observation,
                            root_seat=root_seat,
                            particle=particle,
                            branch=branch,
                            evaluation_cache=evaluation_cache,
                            hidden_or_chance_scenario="|".join(scenario_parts),
                        )
                        leaf_evaluations += int(forward_issued)
                        break
                if value is None:
                    # A time/depth interruption does not count as a completed
                    # simulation and never contributes a partial backup.
                    continue
                self._backup(nodes, edges, value)
                sims_run += 1
                value_backups += 1
                max_simulator_search_depth = max(
                    max_simulator_search_depth, simulator_steps
                )
                if simulator_steps >= 2:
                    multi_step_simulations += 1
                # PUCT already allocates visits in proportion to the frozen
                # policy priors, so a dominant 95% line naturally receives
                # almost all early work while a 3% challenger is still given
                # a chance to disprove it.  Stop early only when a complete
                # root action (not a factorized prefix) has remained selected
                # and has both a decisive visit share and a backed-up Q gap.
                if not root.factorized and root.edges:
                    ranked = sorted(
                        root.edges,
                        key=lambda candidate: (
                            candidate.visit,
                            candidate.prior,
                            tuple(-item for item in candidate.action),
                        ),
                        reverse=True,
                    )
                    best = ranked[0]
                    best_action = tuple(best.action)
                    if best_action == convergence_best_action:
                        convergence_stable_backups += 1
                    else:
                        convergence_best_action = best_action
                        convergence_stable_backups = 1
                    visit_share = best.visit / max(1, root.visit)
                    second_q = max(
                        (candidate.q() for candidate in ranked[1:]),
                        default=float("-inf"),
                    )
                    q_margin = None if len(ranked) == 1 else best.q() - second_q
                    single_legal_action = len(ranked) == 1 and sims_run >= 1
                    stable_dominant_action = (
                        sims_run >= self.convergence_min_sims
                        and convergence_stable_backups
                        >= self.convergence_stable_sims
                        and visit_share >= self.convergence_visit_share
                        and q_margin is not None
                        and q_margin >= self.convergence_q_margin
                    )
                    if single_legal_action or stable_dominant_action:
                        root_action_stable = True
                        root_stability_receipt = {
                            # These explicit facts are consumed by the
                            # full-turn controller before it labels a search
                            # as convergence-stopped.  The receipt is emitted
                            # only immediately after ``_backup`` above.
                            "stable_root_convergence": True,
                            "selected_action": list(best_action),
                            "selected_action_legal": True,
                            "selected_action_fully_backed_up": bool(
                                best.visit >= 1 and value_backups >= 1
                            ),
                            "selected_action_visit_count": int(best.visit),
                            "selected_action_completed_backups": int(best.visit),
                            "completed_backups": sims_run,
                            "stable_backups": convergence_stable_backups,
                            "visit_share": visit_share,
                            "q_margin": q_margin,
                            "single_legal_action": single_legal_action,
                            "root_value_or_visit_stability_evidence": {
                                "root_visit_count": int(root.visit),
                                "selected_action_visit_count": int(best.visit),
                                "visit_share": visit_share,
                                "q_margin": q_margin,
                                "stable_backups": convergence_stable_backups,
                            },
                        }
                        break
            finally:
                search_backend.search_end()

        elapsed = time.perf_counter() - started
        if clock is not None:
            clock.consume(elapsed)
        if sims_run < self.min_trusted_sims:
            raise TrustedSearchBudgetExhausted(
                f"insufficient trusted belief simulations: completed "
                f"{sims_run}/{sims_plan} within {move_budget:.3f}s"
            )
        visits = [edge.visit for edge in root.edges]
        priors = [edge.prior for edge in root.edges]
        combos = [edge.action for edge in root.edges]
        factorized_stages: list[dict[str, Any]] = []
        selected_action: Optional[list[int]] = None
        if root.factorized:
            stage_node = root
            while True:
                stage_combos = [list(edge.action) for edge in stage_node.edges]
                stage_visits = [int(edge.visit) for edge in stage_node.edges]
                stage_priors = [float(edge.prior) for edge in stage_node.edges]
                if not stage_combos:
                    raise RuntimeError("factorized action node has no candidates")
                if sum(stage_visits) > 0:
                    stage_policy = build_search_target(
                        stage_combos,
                        stage_visits,
                        stage_node.q(),
                        prior=stage_priors,
                        temperature=temperature,
                    ).policy
                    selected_index = max(
                        range(len(stage_combos)),
                        key=lambda index: (
                            stage_visits[index],
                            stage_priors[index],
                            -index,
                        ),
                    )
                else:
                    prior_total = sum(stage_priors)
                    stage_policy = (
                        [prior / prior_total for prior in stage_priors]
                        if prior_total > 0
                        else [1.0 / len(stage_priors)] * len(stage_priors)
                    )
                    selected_index = max(
                        range(len(stage_combos)),
                        key=lambda index: (stage_priors[index], -index),
                    )
                candidate = stage_combos[selected_index]
                factorized_stages.append(
                    {
                        "action_combos": stage_combos,
                        "policy": stage_policy,
                        "visits": stage_visits,
                        "selected_index": selected_index,
                    }
                )
                if self._factorized_action_complete(
                    obs_dict,
                    stage_node.action_prefix,
                    candidate,
                ):
                    selected_action = candidate
                    break
                key = self._factorized_key(root.fingerprint, candidate)
                selected_edge = stage_node.edges[selected_index]
                child = selected_edge.outcomes.get(key)
                if child is None:
                    child = BeliefNode(
                        fingerprint=key,
                        actor=root_seat,
                        depth=stage_node.depth + 1,
                    )
                    selected_edge.outcomes[key] = child
                if not child.network_evaluated:
                    _factorized_value, forward_issued = self._evaluate_node_cached(
                        child,
                        obs_dict,
                        root_seat=root_seat,
                        particle=None,
                        branch=root_branch,
                        action_prefix=candidate,
                        evaluation_cache=evaluation_cache,
                        hidden_or_chance_scenario=(
                            "public-root|factorized-prefix:"
                            + ",".join(str(index) for index in candidate)
                        ),
                    )
                    leaf_evaluations += int(forward_issued)
                stage_node = child
        nodes = self._all_nodes(root)
        expanded = [node for node in nodes if node.network_evaluated]
        inference = self._telemetry_since(telemetry_marker)
        action = (
            selected_action
            if selected_action is not None
            else select_by_visits(combos, visits)
        )
        selected_action_tuple = tuple(int(index) for index in action)
        if root.factorized:
            selected_action_legal = bool(
                selected_action is not None and factorized_stages
            )
            selected_stage_visits = [
                int(stage["visits"][int(stage["selected_index"])])
                for stage in factorized_stages
            ]
            selected_action_visit_count = (
                min(selected_stage_visits) if selected_stage_visits else 0
            )
        else:
            selected_root_edge = next(
                (
                    edge
                    for edge in root.edges
                    if tuple(int(index) for index in edge.action)
                    == selected_action_tuple
                ),
                None,
            )
            selected_action_visit_count = (
                int(selected_root_edge.visit)
                if selected_root_edge is not None
                else 0
            )
            selected_action_legal = selected_root_edge is not None
        selected_action_fully_backed_up = bool(
            selected_action_legal
            and selected_action_visit_count >= 1
            and value_backups >= 1
        )
        principal_continuation = self._principal_continuation_by_fingerprint(
            root,
            root_seat=root_seat,
            selected_action=action,
        )
        diagnostics = {
            "sims_run": sims_run,
            "sims_planned": sims_plan,
            "sims_requested": int(
                requested_max_sims
                if requested_max_sims is not None
                else sims_plan
            ),
            "adaptive_sim_cap": sims_plan,
            "stop_reason": (
                "converged_root_action"
                if root_action_stable
                else (
                    "emergency_simulation_safety_ceiling"
                    if sims_run >= sims_plan
                    else "valid_move_time_budget"
                )
            ),
            "elapsed_s": elapsed,
            "move_budget_s": move_budget,
            "sims_per_s": sims_run / max(elapsed, 1e-9),
            "unique_nodes": len(nodes),
            "unique_expanded_nodes": len(expanded),
            "max_depth": max((node.depth for node in nodes), default=0),
            "mean_depth": sum(node.depth for node in nodes) / max(len(nodes), 1),
            "mean_branching": sum(len(node.edges) for node in expanded)
            / max(len(expanded), 1),
            "leaf_evaluations": leaf_evaluations,
            "frozen_model_forwards_issued": leaf_evaluations,
            "simulator_transitions": simulator_transitions,
            "deterministic_successor_expansions": (
                deterministic_successor_expansions
            ),
            "exact_terminal_results_seen": exact_terminal_results_seen,
            "value_backups": value_backups,
            "max_simulator_search_depth": max_simulator_search_depth,
            "multi_step_simulations": multi_step_simulations,
            "root_action_stable": root_action_stable,
            "root_stability_receipt": root_stability_receipt,
            "selected_action": list(selected_action_tuple),
            "selected_action_legal": selected_action_legal,
            "selected_action_fully_backed_up": selected_action_fully_backed_up,
            "selected_action_visit_count": selected_action_visit_count,
            "selected_action_completed_backups": selected_action_visit_count,
            "completed_backups": value_backups,
            "convergence_min_sims": self.convergence_min_sims,
            "convergence_stable_sims": self.convergence_stable_sims,
            "convergence_visit_share": self.convergence_visit_share,
            "convergence_q_margin": self.convergence_q_margin,
            "unique_deterministic_state_evaluation_keys": len(
                evaluation_cache.deterministic_state_keys
            ),
            "deterministic_state_model_evaluation_cache_hits": (
                evaluation_cache.hits
            ),
            "deterministic_state_model_evaluation_cache_misses": (
                evaluation_cache.misses
            ),
            "deterministic_state_model_evaluation_cache_entries": len(
                evaluation_cache.entries
            ),
            "hidden_or_chance_partitioned_evaluation_keys": len(
                evaluation_cache.hidden_or_chance_partitioned_keys
            ),
            "one_model_evaluation_per_unique_deterministic_state_key_verified": (
                evaluation_cache.misses == len(evaluation_cache.entries)
            ),
            "evaluation_cache_scope": "one_search_call_private_world_partitioned",
            "native_complete_semantic_state_identity_available": False,
            "native_actions_commute_certificate_available": False,
            "transposition_merges_attempted": 0,
            "transposition_merges_accepted": 0,
            "transposition_merges_rejected": 0,
            "transposition_merge_rejection_reasons": {
                "native_complete_semantic_state_identity_unavailable": 0
            },
            "transposition_model_evaluation_savings": 0,
            "chance_samples": chance_samples,
            "finite_chance_enumerations": finite_chance_enumerations,
            "finite_chance_outcomes_enumerated": (
                finite_chance_outcomes_enumerated
            ),
            "finite_chance_weighted_backup_count": (
                finite_chance_weighted_backup_count
            ),
            "finite_chance_forced_successor_transitions": (
                finite_chance_forced_successor_transitions
            ),
            "unforceable_chance_boundary_nodes": (
                unforceable_chance_boundary_nodes
            ),
            "unforceable_chance_boundary_leaf_evaluations": (
                unforceable_chance_boundary_leaf_evaluations
            ),
            "unforceable_chance_boundary_reasons": dict(
                sorted(unforceable_chance_boundary_reasons.items())
            ),
            "private_unforceable_chance_samples_prohibited": True,
            "seed_hunting_or_pre_randomization_prohibited": True,
            "exact_finite_chance_capability_configured": (
                getattr(self, "exact_finite_chance_capability", None) is not None
                or getattr(self, "finite_chance_capability", None) is not None
            ),
            "sampled_unforceable_chance_nodes": (
                sampled_unforceable_chance_nodes
            ),
            "sampled_unforceable_chance_reasons": dict(
                sorted(sampled_unforceable_chance_reasons.items())
            ),
            "particle_decks": len(particle_decks),
            "particle_bank_size": len(particle_bank),
            "particle_support_modes": particle_support_modes,
            "particle_support_repairs": particle_support_repairs,
            "particles_sampled": particle_attempts,
            "unique_particles": len(particle_states),
            "root_visits": root.visit,
            "n_options": len(combos),
            "complete_ordered_action_count": root.total_action_count,
            "root_information_state_fingerprint": root.fingerprint,
            "trusted": True,
            "search_semantics": "public_history_root_sampled_information_set_mcts",
            "leaf_evaluator": self.leaf_evaluator_source,
            "leaf_evaluator_checkpoint_digest": (
                self.leaf_evaluator_checkpoint_digest
            ),
            "belief_mode": "anonymous_empirical_deck_particles",
            "chance_mode": (
                "capability_gated_exact_finite_enumeration_with_pre_random_boundaries"
                if finite_chance_enumerations and unforceable_chance_boundary_nodes
                else (
                    "capability_gated_exact_finite_enumeration"
                    if finite_chance_enumerations
                    else (
                        "unforceable_random_pre_boundary_leaf"
                        if unforceable_chance_boundary_nodes
                        else "no_chance_encountered"
                    )
                )
            ),
            # r219 permits a narrow local exact expectation only for a sealed
            # forced finite distribution.  This public-history root-sampled
            # implementation remains explicitly non-r207-exact overall.
            "r207_exact_finite_chance_probability_weighted_expectation_claimed": (
                False
            ),
            "principal_continuation_by_fingerprint": principal_continuation,
            "principal_continuation_cache_candidates": len(
                principal_continuation
            ),
            "principal_continuation_derivation": (
                "single_outcome_no_sampled_chance_root_sampled_approximate"
            ),
            "tree_reuse": False,
            "history_mode": "branch_local_actor_history",
            "action_space_mode": (
                "exact_autoregressive_hierarchical"
                if root.factorized
                else "complete_materialized"
            ),
            "factorized_stages": factorized_stages,
            "matchup_adapter_shadow": (
                self.matchup_shadow_router.audit.snapshot(include_events=False)
                if self.matchup_shadow_router is not None
                else None
            ),
            **inference,
        }
        # Scope B (Blackwell Hammer): optional root-only value bias from lethal
        # head. Documented choice — bias value aux at root; do not rewrite
        # board bags or leaf policy. Disabled on core / when heads warm-started.
        root_value = float(root.q())
        if neural_priors.lethal_threat_logit is not None:
            bias = root_value_bias_from_lethal(
                torch.tensor(neural_priors.lethal_threat_logit)
            )
            root_value = max(-1.0, min(1.0, root_value + bias))
            diagnostics["blackwell_lethal_value_bias"] = bias
            diagnostics["blackwell_prize_race"] = (
                list(neural_priors.prize_race)
                if neural_priors.prize_race is not None
                else None
            )
        target = build_search_target(
            combos,
            visits,
            root_value,
            prior=priors,
            temperature=temperature,
            diagnostics=diagnostics,
        )
        return MCTSResult(
            select=action,
            target=target,
            sims_run=sims_run,
            elapsed_s=elapsed,
        )

    @staticmethod
    def _all_nodes(root: BeliefNode) -> list[BeliefNode]:
        seen: set[int] = set()
        stack = [root]
        out: list[BeliefNode] = []
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            out.append(node)
            for edge in node.edges:
                stack.extend(edge.outcomes.values())
        return out

    @staticmethod
    def _principal_edge(node: BeliefNode) -> Optional[BeliefEdge]:
        """Choose one fully backed-up deterministic principal edge.

        This is a cache-candidate extractor only, never an action selector for
        the real game.  The controller still validates the public observation,
        complete legal order, and selected action immediately before dispatch.
        """

        if not node.edges:
            return None
        index, edge = max(
            enumerate(node.edges),
            key=lambda item: (
                item[1].visit,
                item[1].prior,
                -item[0],
            ),
        )
        del index
        return edge if edge.visit >= 1 else None

    @staticmethod
    def _only_deterministic_child(edge: BeliefEdge) -> Optional[BeliefNode]:
        if edge.saw_sampled_or_opaque_chance or len(edge.outcomes) != 1:
            return None
        return next(iter(edge.outcomes.values()))

    def _principal_continuation_by_fingerprint(
        self,
        root: BeliefNode,
        *,
        root_seat: int,
        selected_action: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Expose one conservative cache candidate for a later own decision.

        Root-sampled search does not have native complete semantic-state
        attestation, so this output makes no transposition or exact-chance
        claim.  It is emitted only along a single-outcome, no-sampled-chance
        principal path.  A runtime may reuse it solely after a fresh public
        fingerprint/legal-order/action check; otherwise it is discarded and
        the shared-turn controller can re-search from its residual budget.
        """

        if root.factorized:
            return []
        root_edge = next(
            (
                edge
                for edge in root.edges
                if tuple(edge.action) == tuple(selected_action)
            ),
            None,
        )
        if root_edge is None or root_edge.visit < 1:
            return []
        node = self._only_deterministic_child(root_edge)
        if node is None:
            return []
        # A bounded principal walk may cross one or more fully deterministic
        # opponent decisions before returning to our next actual decision.
        # It deliberately never skips a chance/information boundary.
        for principal_depth in range(1, 33):
            if node.actor == root_seat:
                if node.factorized or not node.network_evaluated:
                    return []
                edge = self._principal_edge(node)
                if edge is None:
                    return []
                legal_actions = [list(candidate.action) for candidate in node.edges]
                if not legal_actions or list(edge.action) not in legal_actions:
                    return []
                return [
                    {
                        "expected_public_observation_fingerprint": node.fingerprint,
                        "legal_actions": legal_actions,
                        "selected_action": list(edge.action),
                        "deterministic": True,
                        "selected_action_fully_backed_up": True,
                        "principal_path_depth": principal_depth,
                        "derivation": (
                            "single_outcome_no_sampled_chance_principal_path_"
                            "root_sampled_approximate"
                        ),
                    }
                ]
            edge = self._principal_edge(node)
            if edge is None:
                return []
            node = self._only_deterministic_child(edge)
            if node is None:
                return []
        return []

    def target_provenance(
        self,
        *,
        max_sims: int,
        move_time_s: float,
    ) -> dict[str, Any]:
        return {
            "checkpoint_digest": self.checkpoint_digest,
            "model_generation": self.model_generation,
            "search_config": {
                "algorithm": "public_history_root_sampled_information_set_mcts",
                "max_sims": int(max_sims),
                "min_trusted_sims": self.min_trusted_sims,
                "move_time_s": float(move_time_s),
                "puct_c": self.puct_c,
                "max_depth": self.max_depth,
                "convergence_min_sims": self.convergence_min_sims,
                "convergence_stable_sims": self.convergence_stable_sims,
                "convergence_visit_share": self.convergence_visit_share,
                "convergence_q_margin": self.convergence_q_margin,
                "adaptive_sequential_updates": True,
                "cross_game_batching_only": True,
                "tree_reuse": False,
                "particle_count": self.particle_count,
                "action_space": "exact_autoregressive_hierarchical_when_large",
                "leaf_evaluator": self.leaf_evaluator_source,
                "leaf_evaluator_checkpoint_digest": (
                    self.leaf_evaluator_checkpoint_digest
                ),
            },
            "belief_config": self.posterior.config,
            "simulator_version": self._simulator_version,
        }
