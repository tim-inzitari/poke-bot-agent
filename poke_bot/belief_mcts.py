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
from typing import Any, Callable, Optional, Sequence

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
from .heuristics_registry import (
    apply_prior_logit_bias,
    resolve_heuristics,
)
from .mcts import GameClock, MCTSResult
from .model import TemporalCabtTransformer, card_prior_logits_or_uniform
from .opening_budget import (
    clarity_caps_to_floor,
    observation_turn,
    scale_opening_budgets,
    visit_stop_triggered,
)
from .replay_import import assert_info_set as assert_public_info_set
from .search_targets import build_search_target, select_by_visits


@dataclass
class BeliefEdge:
    action: list[int]
    prior: float
    visit: int = 0
    total: float = 0.0
    outcomes: dict[str, "BeliefNode"] = field(default_factory=dict)

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

    def q(self) -> float:
        return self.total / self.visit if self.visit else 0.0


@dataclass
class _BranchHistory:
    boards: dict[int, list[features.SparseVector]]
    previous_actions: dict[int, list[Optional[features.SparseVector]]]
    last_action: dict[int, Optional[features.SparseVector]]


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
        max_depth: int = 256,
        max_context: Optional[int] = None,
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
        else:
            self.leaf_eval = leaf_backend
        self.puct_c = float(puct_c if puct_c is not None else config.SEARCH.puct_c)
        self.rng = rng or random.Random()
        self.min_trusted_sims = int(min_trusted_sims)
        self.particle_count = max(2, int(particle_count))
        self.max_depth = int(max_depth)
        self.max_context = int(
            max_context
            if max_context is not None
            else getattr(model, "max_context", config.MODEL.max_context)
        )
        self._simulator_version = simulator_version()

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
        raw = _raw_observation(obs)
        prefix = [int(index) for index in action_prefix]
        node.total_action_count = features.ordered_action_count(raw)
        if prefix:
            combos = features.factorized_action_candidates(raw, prefix)
            node.factorized = True
        else:
            try:
                combos = features.enumerate_action_combos(raw)
            except features.ActionSpaceTooLarge:
                combos = features.factorized_action_candidates(raw, [])
                node.factorized = True
        node.action_prefix = tuple(prefix)
        packet = self._packet(
            obs,
            root_seat=root_seat,
            particle=particle,
            branch=branch,
            combos=[list(combo) for combo in combos],
        )
        output = self.leaf_eval([packet])
        if len(output) != 1:
            raise RuntimeError("leaf backend returned wrong response count")
        evaluated = output[0]
        if evaluated.combos != list(combos):
            raise RuntimeError("leaf backend changed complete legal action ordering")
        if len(evaluated.priors) != len(combos):
            raise RuntimeError("leaf prior/action count mismatch")
        priors = [float(prior) for prior in evaluated.priors]
        heuristics = resolve_heuristics(deck_card_ids=self.own_deck)
        bias = heuristics.prior_logit_bias(
            raw, [list(combo) for combo in combos]
        )
        if any(abs(float(b)) > 0.0 for b in bias):
            priors = apply_prior_logit_bias(priors, bias)
        node.edges = [
            BeliefEdge(action=list(combo), prior=float(prior))
            for combo, prior in zip(combos, priors)
        ]
        node.network_evaluated = True
        return float(evaluated.value)

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

    def _sample_chance_until_decision(
        self,
        search_state,
    ) -> tuple[Any, int]:
        samples = 0
        state = search_state
        while (
            self._terminal_value(state.observation, 0) is None
            and is_explicit_chance(state.observation)
        ):
            combos = features.enumerate_action_combos(
                _raw_observation(state.observation)
            )
            if not combos:
                raise RuntimeError("explicit chance node has no outcomes")
            # COIN_HEAD exposes equiprobable YES/NO. This random choice is never
            # scored by PUCT or attributed to either player's strategy.
            action = list(self.rng.choice(list(combos)))
            state = cg_env.search_step(state.searchId, action)
            samples += 1
        return state, samples

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
        turn = observation_turn(obs_dict)
        opening_sims, opening_move, opening_applied = scale_opening_budgets(
            turn=turn,
            requested_sims=requested_sims,
            move_time_s=move_budget,
            min_trusted_sims=self.min_trusted_sims,
        )
        if opening_applied:
            move_budget = opening_move
        # Keep 128 as a hard trust floor, but shed optional ramped simulations
        # when the per-game clock grants less than the configured move slice.
        ratio = min(1.0, move_budget / configured_budget) if configured_budget > 0 else 1.0
        adaptive_sims = max(
            self.min_trusted_sims,
            min(
                opening_sims,
                int(math.ceil(opening_sims * ratio)),
            ),
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
                opening_budget_applied=opening_applied,
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
        opening_budget_applied: bool = False,
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
        self._evaluate_node(
            root,
            obs_dict,
            root_seat=root_seat,
            particle=None,
            branch=root_branch,
        )
        leaf_evaluations = 1
        clarity_prior_stop = False
        visit_stop = False
        root_priors = [edge.prior for edge in root.edges]
        sims_plan, clarity_prior_stop = clarity_caps_to_floor(
            root_priors,
            min_trusted_sims=self.min_trusted_sims,
            current_plan=sims_plan,
        )
        heuristics_mod = resolve_heuristics(deck_card_ids=self.own_deck)
        heuristic_arch = str(
            getattr(heuristics_mod, "describe", lambda: "unknown")()
        )
        chance_samples = 0
        sims_run = 0
        particle_decks: set[str] = set()
        particle_states: set[str] = set()
        particle_attempts = 0
        # Root-only NN reweight of the empirical posterior / hand fill.
        neural_priors = self._root_neural_priors(
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
        stop_reason = "simulation_target"
        # Leave a tiny reserve so we do not enqueue a leaf RPC that is already
        # doomed to ``RemoteLeafTimeout: ... after 0.0Xs`` under the move deadline.
        leaf_reserve_s = 0.05
        while (
            sims_run < sims_plan
            and time.perf_counter() - started < move_budget - leaf_reserve_s
        ):
            if visit_stop_triggered(
                [edge.visit for edge in root.edges],
                sims_run=sims_run,
                sims_plan=sims_plan,
                min_trusted_sims=self.min_trusted_sims,
                elapsed_s=time.perf_counter() - started,
                move_budget_s=move_budget,
            ):
                visit_stop = True
                stop_reason = "visit_stop"
                break
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
            search_state = cg_env.search_begin(
                obs_dict, particle.search_inputs, manual_coin=True
            )
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
            nodes = [root]
            edges: list[BeliefEdge] = []
            current = root
            value: Optional[float] = None
            try:
                for depth in range(1, self.max_depth + 1):
                    if time.perf_counter() - started >= move_budget - leaf_reserve_s:
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
                            try:
                                value = self._evaluate_node(
                                    child,
                                    search_state.observation,
                                    root_seat=root_seat,
                                    particle=particle,
                                    branch=branch,
                                    action_prefix=edge.action,
                                )
                            except TimeoutError:
                                # Move-deadline-bounded leaf RPC — treat as a
                                # time interruption, not a hard resource fault.
                                value = None
                                break
                            leaf_evaluations += 1
                            break
                        continue
                    branch.last_action[current.actor] = features.build_option_tokens(
                        _raw_observation(search_state.observation), [edge.action]
                    )
                    search_state = cg_env.search_step(
                        search_state.searchId, edge.action
                    )
                    search_state, sampled = self._sample_chance_until_decision(
                        search_state
                    )
                    chance_samples += sampled
                    terminal = self._terminal_value(
                        search_state.observation, root_seat
                    )
                    if terminal is not None:
                        value = terminal
                        break
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
                    nodes.append(child)
                    current = child
                    if not child.network_evaluated:
                        try:
                            value = self._evaluate_node(
                                child,
                                search_state.observation,
                                root_seat=root_seat,
                                particle=particle,
                                branch=branch,
                            )
                        except TimeoutError:
                            value = None
                            break
                        leaf_evaluations += 1
                        break
                if value is None:
                    # A time/depth interruption does not count as a completed
                    # simulation and never contributes a partial backup.
                    continue
                self._backup(nodes, edges, value)
                sims_run += 1
            finally:
                cg_env.search_end()

        elapsed = time.perf_counter() - started
        if clock is not None:
            clock.consume(elapsed)
        if sims_run < self.min_trusted_sims:
            raise TrustedSearchBudgetExhausted(
                f"insufficient trusted belief simulations: completed "
                f"{sims_run}/{sims_plan} within {move_budget:.3f}s"
            )
        if not visit_stop and sims_run < sims_plan:
            stop_reason = "valid_move_time_budget"
        elif not visit_stop:
            stop_reason = "simulation_target"
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
                    self._evaluate_node(
                        child,
                        obs_dict,
                        root_seat=root_seat,
                        particle=None,
                        branch=root_branch,
                        action_prefix=candidate,
                    )
                    leaf_evaluations += 1
                stage_node = child
        nodes = self._all_nodes(root)
        expanded = [node for node in nodes if node.network_evaluated]
        inference = self._telemetry_since(telemetry_marker)
        diagnostics = {
            "sims_run": sims_run,
            "sims_planned": sims_plan,
            "sims_requested": int(
                requested_max_sims
                if requested_max_sims is not None
                else sims_plan
            ),
            "adaptive_sim_cap": sims_plan,
            "stop_reason": stop_reason,
            "opening_budget": bool(opening_budget_applied),
            "clarity_prior_stop": bool(clarity_prior_stop),
            "visit_stop": bool(visit_stop),
            "heuristic_arch": heuristic_arch,
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
            "chance_samples": chance_samples,
            "particle_decks": len(particle_decks),
            "particle_bank_size": len(particle_bank),
            "particle_support_modes": particle_support_modes,
            "particle_support_repairs": particle_support_repairs,
            "particles_sampled": particle_attempts,
            "unique_particles": len(particle_states),
            "root_visits": root.visit,
            "n_options": len(combos),
            "complete_ordered_action_count": root.total_action_count,
            "trusted": True,
            "search_semantics": "public_history_root_sampled_information_set_mcts",
            "belief_mode": "anonymous_empirical_deck_particles",
            "chance_mode": "explicit_uniform_coin_sampling",
            "tree_reuse": False,
            "history_mode": "branch_local_actor_history",
            "action_space_mode": (
                "exact_autoregressive_hierarchical"
                if root.factorized
                else "complete_materialized"
            ),
            "factorized_stages": factorized_stages,
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
        action = (
            selected_action
            if selected_action is not None
            else select_by_visits(combos, visits)
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
                "adaptive_sequential_updates": True,
                "cross_game_batching_only": True,
                "tree_reuse": False,
                "particle_count": self.particle_count,
                "action_space": "exact_autoregressive_hierarchical_when_large",
            },
            "belief_config": self.posterior.config,
            "simulator_version": self._simulator_version,
        }

