"""AlphaZero PUCT MCTS on the official competition Search API.

Tree edges are legal option-index combos from ``obs.select``. Expansion uses
``search_begin`` / ``search_step`` / ``search_release`` / ``search_end``.

Hard rules:
  - Policy priors and leaf values come from the info-set Temporal Transformer
    (no privileged opponent hand / prizes inside search).
  - Time-budgeted (~600s/game via :class:`~poke_bot.config.SearchConfig`);
    dynamic thinking allocates more sims to complex positions.
  - Visit-count improved policy targets via :mod:`poke_bot.search_targets`.
  - GPU batching is on **network eval only** (see :mod:`poke_bot.batched_infer`);
    each tree still owns its own ``search_begin`` / ``search_step``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from . import cg_env, config, features
from .batched_infer import BatchedLeafServer, LeafPacket, forward_leaf_batch
from .model import TemporalCabtTransformer, TemporalKVCache
from .search_targets import SearchTarget, build_search_target, select_by_visits


@dataclass
class Child:
    select: list[int]
    prior: float
    node: Optional["Node"] = None
    #: Virtual-loss count while a leaf under this edge is in a pending GPU batch.
    virtual_loss: int = 0


@dataclass
class Node:
    state: object  # cg.api.SearchState
    parent: Optional["Node"] = None
    children: list[Child] = field(default_factory=list)
    visit: int = 0
    total: float = 0.0
    value: float = 0.0  # leaf/network value from root seat perspective
    is_terminal: bool = False
    evaluated: bool = False

    def q(self) -> float:
        # Treat pending virtual loss as visits with value 0 so siblings diversify.
        v = self.visit
        if v <= 0:
            return 0.0
        return self.total / v

    def backprop(self, value: float) -> None:
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


@dataclass
class MCTSResult:
    select: list[int]
    target: SearchTarget
    sims_run: int
    elapsed_s: float


@dataclass
class GameClock:
    """Shared per-game think budget (~600s)."""

    total_s: float = field(default_factory=lambda: config.SEARCH.game_time_budget_s)
    remaining_s: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_s = float(self.total_s)

    def consume(self, used: float) -> None:
        self.remaining_s = max(0.0, self.remaining_s - used)


class LeafEvaluator:
    """Info-set leaf evaluation; batches when several leaves share no KV dependency."""

    def __init__(
        self,
        model: TemporalCabtTransformer,
        your_deck: list[int],
        root_seat: int,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
    ):
        self.model = model
        self.your_deck = your_deck
        self.root_seat = root_seat
        self.device = device or next(model.parameters()).device
        self.batch_size = batch_size or config.SEARCH.leaf_batch_size
        self.model.eval()
        self._server = BatchedLeafServer(
            model, batch_size=self.batch_size
        )

    @torch.no_grad()
    def evaluate_one(
        self,
        obs,
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append_cache: bool = False,
    ) -> tuple[float, list[float], list[list[int]], Optional[TemporalKVCache]]:
        """Return ``(value_root_seat, priors, combos, new_cache)``.

        ``kv_cache`` is accepted for API compatibility; batched leaf path does
        not update KV (search leaves are independent of the live-game cache).
        """
        del kv_cache, append_cache
        packet = LeafPacket(obs=obs, your_deck=self.your_deck, root_seat=self.root_seat)
        out = forward_leaf_batch(self.model, [packet])[0]
        return out.value, out.priors, out.combos, None

    @torch.no_grad()
    def evaluate_batch(
        self,
        obs_list: Sequence[object],
    ) -> list[tuple[float, list[float], list[list[int]]]]:
        """Batched forward when several leaves share no KV dependency."""
        if not obs_list:
            return []
        packets = [
            LeafPacket(obs=o, your_deck=self.your_deck, root_seat=self.root_seat)
            for o in obs_list
        ]
        outs = self._server.evaluate_now(packets)
        return [(o.value, o.priors, o.combos) for o in outs]


def planned_sims(
    n_options: int,
    clock: Optional[GameClock] = None,
    *,
    base_sims: Optional[int] = None,
    move_budget_s: Optional[float] = None,
) -> tuple[int, float]:
    """Decide sim count + move time budget (dynamic thinking)."""
    sc = config.SEARCH
    base = base_sims if base_sims is not None else sc.sims_per_move
    move_t = move_budget_s if move_budget_s is not None else sc.move_time_budget_s
    if clock is not None:
        move_t = min(move_t, max(0.05, clock.remaining_s))

    sims = base
    if n_options >= sc.complex_option_threshold:
        sims = int(math.ceil(sims * sc.complex_sims_mult))
    sims = max(sc.min_sims, sims)
    return sims, move_t


@dataclass
class _PendingLeaf:
    """Leaf awaiting GPU eval (virtual-loss already applied on the path)."""

    parent: Optional[Node]
    child: Optional[Child]  # edge from parent; None for root expand
    search_state: object
    path_edges: list[Child]  # edges that received +1 virtual_loss


class MCTS:
    """Time-budgeted AlphaZero MCTS over one decision (GPU-batched leaves)."""

    def __init__(
        self,
        model: TemporalCabtTransformer,
        your_deck: list[int],
        *,
        device: Optional[torch.device] = None,
        puct_c: Optional[float] = None,
        opponent_deck_guess: Optional[list[int]] = None,
        leaf_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.your_deck = your_deck
        self.opponent_deck_guess = opponent_deck_guess or your_deck
        self.device = device or next(model.parameters()).device
        self.puct_c = puct_c if puct_c is not None else config.SEARCH.puct_c
        self.leaf_batch_size = (
            leaf_batch_size
            if leaf_batch_size is not None
            else config.SEARCH.leaf_batch_size
        )

    def _terminal_value(self, obs, root_seat: int) -> Optional[float]:
        state = obs.current if hasattr(obs, "current") else None
        if state is None:
            return None
        if state.result < 0:
            return None
        if state.result == 2:
            return 0.0
        if state.result == root_seat:
            return 1.0
        return -1.0

    def _apply_eval(
        self,
        node: Node,
        value: float,
        priors: list[float],
        combos: list[list[int]],
    ) -> None:
        node.value = value
        node.evaluated = True
        node.backprop(value)
        for combo, prior in zip(combos, priors):
            node.children.append(Child(select=list(combo), prior=float(prior)))
        if not node.children:
            node.is_terminal = True

    def _expand_terminal_or_pending(
        self,
        parent: Optional[Node],
        search_state,
        root_seat: int,
        path_edges: list[Child],
    ) -> tuple[Optional[Node], Optional[_PendingLeaf]]:
        """Return (fully_resolved_node, pending_leaf). One of the two is set."""
        obs = search_state.observation
        node = Node(state=search_state, parent=parent)
        term = self._terminal_value(obs, root_seat)
        if term is not None:
            node.is_terminal = True
            node.evaluated = True
            node.value = term
            node.backprop(term)
            return node, None
        return None, _PendingLeaf(
            parent=parent,
            child=None,
            search_state=search_state,
            path_edges=list(path_edges),
        )

    def _materialize_pending(
        self,
        pending: _PendingLeaf,
        packet: LeafPacket,
    ) -> Node:
        parent = pending.parent
        node = Node(state=pending.search_state, parent=parent)
        self._apply_eval(node, packet.value, packet.priors, packet.combos)
        if pending.child is not None:
            pending.child.node = node
        for edge in pending.path_edges:
            edge.virtual_loss = max(0, edge.virtual_loss - 1)
        return node

    def _select_child(self, node: Node) -> Child:
        best: Optional[Child] = None
        best_score = -1e18
        # Effective visit includes virtual losses on edges.
        parent_visit = node.visit + sum(c.virtual_loss for c in node.children)
        c_puct = self.puct_c * math.sqrt(max(parent_visit, 1))
        for child in node.children:
            if child.node is None:
                q = 0.0
                visit = child.virtual_loss
            else:
                q = child.node.q()
                visit = child.node.visit + child.virtual_loss
            u = c_puct * child.prior / (1 + visit)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        assert best is not None
        return best

    def _select_unevaluated_leaf(
        self, root: Node, root_seat: int
    ) -> tuple[Optional[Node], Optional[_PendingLeaf]]:
        """Walk PUCT to an unevaluated edge; apply virtual loss; return pending."""
        current = root
        path_edges: list[Child] = []
        while True:
            if current.is_terminal or not current.children:
                current.backprop(current.value)
                return current, None
            path_child = self._select_child(current)
            if path_child.node is None:
                path_child.virtual_loss += 1
                path_edges.append(path_child)
                child_state = cg_env.search_step(
                    current.state.searchId, path_child.select
                )
                resolved, pending = self._expand_terminal_or_pending(
                    current, child_state, root_seat, path_edges
                )
                if resolved is not None:
                    path_child.node = resolved
                    path_child.virtual_loss = max(0, path_child.virtual_loss - 1)
                    return resolved, None
                assert pending is not None
                pending.child = path_child
                return None, pending
            path_child.virtual_loss += 1
            path_edges.append(path_child)
            current = path_child.node
            if current.is_terminal:
                # Undo virtual loss along path; terminal already has real visits.
                for edge in path_edges:
                    edge.virtual_loss = max(0, edge.virtual_loss - 1)
                current.backprop(current.value)
                return current, None

    def _flush_pending(
        self,
        evaluator: LeafEvaluator,
        pending_list: list[_PendingLeaf],
    ) -> list[Node]:
        """Evaluate pending leaves in one GPU batch; return materialized nodes."""
        if not pending_list:
            return []
        packets = [
            LeafPacket(
                obs=p.search_state.observation,
                your_deck=evaluator.your_deck,
                root_seat=evaluator.root_seat,
            )
            for p in pending_list
        ]
        outs = forward_leaf_batch(self.model, packets)
        nodes: list[Node] = []
        for pend, pkt in zip(pending_list, outs):
            nodes.append(self._materialize_pending(pend, pkt))
        return nodes

    def search(
        self,
        obs_dict: dict,
        *,
        clock: Optional[GameClock] = None,
        max_sims: Optional[int] = None,
        move_time_s: Optional[float] = None,
        temperature: float = 1.0,
    ) -> MCTSResult:
        """Run PUCT search from ``obs_dict``; returns action + visit target."""
        obs = cg_env.to_observation(obs_dict)
        features.assert_info_set(obs)
        if obs.current is None:
            raise ValueError("MCTS requires a post-setup observation (current != None)")
        root_seat = obs.current.yourIndex

        root_combos = features.enumerate_action_combos(obs)
        sims_plan, move_budget = planned_sims(
            len(root_combos), clock, base_sims=max_sims, move_budget_s=move_time_s
        )
        if max_sims is not None:
            sims_plan = max_sims

        evaluator = LeafEvaluator(
            self.model,
            self.your_deck,
            root_seat,
            device=self.device,
            batch_size=self.leaf_batch_size,
        )

        search_inputs = cg_env.build_search_inputs(
            obs_dict, self.your_deck, opponent_deck_guess=self.opponent_deck_guess
        )
        root_state = cg_env.search_begin(obs_dict, search_inputs)
        use_batch = config.SEARCH.leaf_batch_mcts and self.leaf_batch_size > 1

        try:
            t0 = time.perf_counter()
            sims_run = 0

            if use_batch:
                resolved, pending = self._expand_terminal_or_pending(
                    None, root_state, root_seat, []
                )
                if pending is not None:
                    root = self._flush_pending(evaluator, [pending])[0]
                else:
                    assert resolved is not None
                    root = resolved

                pending_buf: list[_PendingLeaf] = []
                while sims_run < sims_plan:
                    if time.perf_counter() - t0 >= move_budget:
                        break
                    if root.is_terminal or not root.children:
                        break

                    # Fill a leaf batch (virtual loss diversifies selections).
                    while (
                        len(pending_buf) < self.leaf_batch_size
                        and sims_run + len(pending_buf) < sims_plan
                        and time.perf_counter() - t0 < move_budget
                    ):
                        _node, pend = self._select_unevaluated_leaf(root, root_seat)
                        if pend is None:
                            sims_run += 1  # terminal / re-visit backprop
                            if root.is_terminal or not root.children:
                                break
                            continue
                        pending_buf.append(pend)

                    if not pending_buf:
                        break
                    sims_run += len(self._flush_pending(evaluator, pending_buf))
                    pending_buf.clear()
            else:
                # Legacy one-leaf-at-a-time path.
                root = Node(state=root_state, parent=None)
                term = self._terminal_value(root_state.observation, root_seat)
                if term is not None:
                    root.is_terminal = True
                    root.evaluated = True
                    root.value = term
                    root.backprop(term)
                else:
                    value, priors, combos, _ = evaluator.evaluate_one(
                        root_state.observation, append_cache=False
                    )
                    self._apply_eval(root, value, priors, combos)

                while sims_run < sims_plan:
                    if time.perf_counter() - t0 >= move_budget:
                        break
                    if root.is_terminal or not root.children:
                        break
                    current = root
                    while True:
                        if current.is_terminal or not current.children:
                            current.backprop(current.value)
                            break
                        path_child = self._select_child(current)
                        if path_child.node is None:
                            child_state = cg_env.search_step(
                                current.state.searchId, path_child.select
                            )
                            child = Node(state=child_state, parent=current)
                            term = self._terminal_value(
                                child_state.observation, root_seat
                            )
                            if term is not None:
                                child.is_terminal = True
                                child.evaluated = True
                                child.value = term
                                child.backprop(term)
                            else:
                                value, priors, combos, _ = evaluator.evaluate_one(
                                    child_state.observation, append_cache=False
                                )
                                self._apply_eval(child, value, priors, combos)
                            path_child.node = child
                            break
                        current = path_child.node
                        if current.is_terminal:
                            current.backprop(current.value)
                            break
                    sims_run += 1

            elapsed = time.perf_counter() - t0
            if clock is not None:
                clock.consume(elapsed)

            visits = [
                (c.node.visit if c.node is not None else 0) for c in root.children
            ]
            priors = [c.prior for c in root.children]
            combos = [c.select for c in root.children]
            root_value = root.q() if root.visit > 0 else root.value
            target = build_search_target(
                combos,
                visits,
                root_value,
                prior=priors,
                temperature=temperature,
                diagnostics={
                    "sims_run": sims_run,
                    "sims_planned": sims_plan,
                    "elapsed_s": elapsed,
                    "move_budget_s": move_budget,
                    "root_visits": root.visit,
                    "n_options": len(combos),
                    "leaf_batch_size": self.leaf_batch_size,
                    "leaf_batch_mcts": use_batch,
                },
            )
            action = select_by_visits(combos, visits) if combos else []
            if not action and root.children:
                best = max(
                    range(len(root.children)), key=lambda i: root.children[i].prior
                )
                action = list(root.children[best].select)
        finally:
            cg_env.search_end()

        return MCTSResult(
            select=action, target=target, sims_run=sims_run, elapsed_s=elapsed
        )


class MultiTreeMCTS:
    """Interleave PUCT across N independent roots; share one GPU leaf batch.

    Each root still gets its own ``search_begin`` / ``search_step`` IDs. Battle
    stepping is *not* batched — only network eval. Useful for reanalyse and for
    ``PARALLEL_GAMES`` snapshot collect.
    """

    def __init__(
        self,
        model: TemporalCabtTransformer,
        your_deck: list[int],
        *,
        device: Optional[torch.device] = None,
        leaf_batch_size: Optional[int] = None,
        puct_c: Optional[float] = None,
    ):
        self.engine = MCTS(
            model,
            your_deck,
            device=device,
            puct_c=puct_c,
            leaf_batch_size=leaf_batch_size,
        )

    def search_many(
        self,
        obs_dicts: Sequence[dict],
        *,
        max_sims: Optional[int] = None,
        move_time_s: Optional[float] = None,
        temperature: float = 1.0,
    ) -> list[MCTSResult]:
        if not obs_dicts:
            return []
        if len(obs_dicts) == 1:
            return [
                self.engine.search(
                    obs_dicts[0],
                    max_sims=max_sims,
                    move_time_s=move_time_s,
                    temperature=temperature,
                )
            ]

        n = len(obs_dicts)
        engines = [self.engine] * n  # shared hyperparams; trees are local below
        del engines

        roots: list[Optional[Node]] = [None] * n
        root_seats: list[int] = []
        sims_plan: list[int] = []
        move_budgets: list[float] = []
        sims_run = [0] * n
        t0 = time.perf_counter()

        # Begin all searches in one libcg session (multi searchId OK).
        for obs_dict in obs_dicts:
            obs = cg_env.to_observation(obs_dict)
            features.assert_info_set(obs)
            if obs.current is None:
                raise ValueError("search_many requires post-setup observations")
            root_seats.append(obs.current.yourIndex)
            combos = features.enumerate_action_combos(obs)
            sp, mb = planned_sims(
                len(combos), None, base_sims=max_sims, move_budget_s=move_time_s
            )
            if max_sims is not None:
                sp = max_sims
            sims_plan.append(sp)
            move_budgets.append(mb)

        evaluator = LeafEvaluator(
            self.engine.model,
            self.engine.your_deck,
            root_seats[0],  # per-tree seat applied when packing packets
            device=self.engine.device,
            batch_size=self.engine.leaf_batch_size,
        )

        try:
            pending_roots: list[_PendingLeaf] = []
            for i, obs_dict in enumerate(obs_dicts):
                si = cg_env.build_search_inputs(
                    obs_dict,
                    self.engine.your_deck,
                    opponent_deck_guess=self.engine.opponent_deck_guess,
                )
                rs = cg_env.search_begin(obs_dict, si)
                resolved, pending = self.engine._expand_terminal_or_pending(
                    None, rs, root_seats[i], []
                )
                if resolved is not None:
                    roots[i] = resolved
                else:
                    assert pending is not None
                    pending_roots.append(pending)

            # Batch-eval all non-terminal roots (fix root_seat per packet).
            if pending_roots:
                packets = [
                    LeafPacket(
                        obs=p.search_state.observation,
                        your_deck=self.engine.your_deck,
                        root_seat=root_seats[j],
                    )
                    for j, p in enumerate(pending_roots)
                ]
                # Map pending_roots index → root slot (only unresolved).
                unresolved_idx = [i for i, r in enumerate(roots) if r is None]
                assert len(unresolved_idx) == len(pending_roots)
                outs = forward_leaf_batch(self.engine.model, packets)
                for j, pend in enumerate(pending_roots):
                    node = self.engine._materialize_pending(pend, outs[j])
                    roots[unresolved_idx[j]] = node

            # Interleaved sims with shared leaf batches.
            pending_buf: list[tuple[int, _PendingLeaf]] = []
            active = set(range(n))
            while active:
                progressed = False
                for i in list(active):
                    root = roots[i]
                    assert root is not None
                    if (
                        sims_run[i] >= sims_plan[i]
                        or time.perf_counter() - t0 >= move_budgets[i]
                        or root.is_terminal
                        or not root.children
                    ):
                        active.discard(i)
                        continue
                    evaluator.root_seat = root_seats[i]
                    _node, pend = self.engine._select_unevaluated_leaf(
                        root, root_seats[i]
                    )
                    if pend is None:
                        sims_run[i] += 1
                        progressed = True
                        continue
                    pending_buf.append((i, pend))
                    progressed = True
                    if len(pending_buf) >= self.engine.leaf_batch_size:
                        self._flush_multi(evaluator, pending_buf, root_seats, sims_run)
                        pending_buf.clear()

                if pending_buf and (
                    len(pending_buf) >= self.engine.leaf_batch_size
                    or not progressed
                    or len(pending_buf) >= len(active)
                ):
                    self._flush_multi(evaluator, pending_buf, root_seats, sims_run)
                    pending_buf.clear()
                elif not progressed:
                    if pending_buf:
                        self._flush_multi(evaluator, pending_buf, root_seats, sims_run)
                        pending_buf.clear()
                    break

            if pending_buf:
                self._flush_multi(evaluator, pending_buf, root_seats, sims_run)

            elapsed = time.perf_counter() - t0
            results: list[MCTSResult] = []
            for i, root in enumerate(roots):
                assert root is not None
                visits = [
                    (c.node.visit if c.node is not None else 0) for c in root.children
                ]
                priors = [c.prior for c in root.children]
                combos = [c.select for c in root.children]
                root_value = root.q() if root.visit > 0 else root.value
                target = build_search_target(
                    combos,
                    visits,
                    root_value,
                    prior=priors,
                    temperature=temperature,
                    diagnostics={
                        "sims_run": sims_run[i],
                        "sims_planned": sims_plan[i],
                        "elapsed_s": elapsed,
                        "multi_tree": True,
                        "n_trees": n,
                    },
                )
                action = select_by_visits(combos, visits) if combos else []
                if not action and root.children:
                    best = max(
                        range(len(root.children)),
                        key=lambda j: root.children[j].prior,
                    )
                    action = list(root.children[best].select)
                results.append(
                    MCTSResult(
                        select=action,
                        target=target,
                        sims_run=sims_run[i],
                        elapsed_s=elapsed,
                    )
                )
            return results
        finally:
            cg_env.search_end()

    def _flush_multi(
        self,
        evaluator: LeafEvaluator,
        pending_buf: list[tuple[int, _PendingLeaf]],
        root_seats: list[int],
        sims_run: list[int],
    ) -> None:
        packets = [
            LeafPacket(
                obs=p.search_state.observation,
                your_deck=self.engine.your_deck,
                root_seat=root_seats[i],
            )
            for i, p in pending_buf
        ]
        outs = forward_leaf_batch(self.engine.model, packets)
        for (i, pend), pkt in zip(pending_buf, outs):
            self.engine._materialize_pending(pend, pkt)
            sims_run[i] += 1


def run_mcts(
    model: TemporalCabtTransformer,
    obs_dict: dict,
    your_deck: list[int],
    *,
    max_sims: int = 8,
    device: Optional[torch.device] = None,
    clock: Optional[GameClock] = None,
) -> MCTSResult:
    """Convenience wrapper for a short search (smoke / greedy-budget calls)."""
    engine = MCTS(model, your_deck, device=device)
    return engine.search(obs_dict, max_sims=max_sims, clock=clock)
