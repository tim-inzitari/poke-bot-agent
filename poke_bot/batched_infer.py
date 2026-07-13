"""GPU-batched policy/value (leaf) inference for MCTS / self-play collect.

CABT ``search_begin`` / ``search_step`` stay per-process and per-tree on CPU.
This module only batches **network** evals: pack board + option SparseVectors,
one ``TemporalCabtTransformer.forward``, scatter values/priors.

See ``outputs/notes/gpu_batched_selfplay.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from . import cg_env, config, features
from .model import TemporalCabtTransformer


def _policy_probs(logits: torch.Tensor) -> list[float]:
    x = logits.detach().float().cpu()
    finite = torch.isfinite(x)
    if not finite.any():
        n = x.numel()
        return [1.0 / max(n, 1)] * n
    masked = x.clone()
    masked[~finite] = float("-inf")
    probs = F.softmax(masked, dim=-1)
    return [float(p) for p in probs.tolist()]


@dataclass
class LeafPacket:
    """One leaf observation ready for (or returned from) a batched forward."""

    obs: object
    your_deck: list[int]
    root_seat: int
    value: float = 0.0
    priors: list[float] = field(default_factory=list)
    combos: list[list[int]] = field(default_factory=list)


@torch.no_grad()
def forward_leaf_batch(
    model: TemporalCabtTransformer,
    packets: Sequence[LeafPacket],
) -> list[LeafPacket]:
    """Vectorized policy/value over ``packets`` (pad options, mask illegal).

    Returns new :class:`LeafPacket` instances with ``value`` / ``priors`` /
    ``combos`` filled (value in each packet's ``root_seat`` perspective).
    """
    if not packets:
        return []
    model.eval()

    boards = []
    opts = []
    combos_list: list[list[list[int]]] = []
    n_opts: list[int] = []
    seats: list[int] = []

    for p in packets:
        features.assert_info_set(p.obs)
        boards.append(features.build_board_tokens(p.obs, p.your_deck))
        combos = features.enumerate_action_combos(p.obs)
        opts.append(features.build_option_tokens(p.obs, combos))
        combos_list.append(combos)
        n_opts.append(len(combos))
        obs_obj = cg_env.to_observation(p.obs) if isinstance(p.obs, dict) else p.obs
        seats.append(
            obs_obj.current.yourIndex if obs_obj.current is not None else p.root_seat
        )

    out = model.forward(
        boards, opts, kv_cache=None, append_cache=False, n_options=n_opts
    )

    results: list[LeafPacket] = []
    for i, p in enumerate(packets):
        combos = combos_list[i]
        logits = out["policy_logits"][i, : len(combos)]
        priors = _policy_probs(logits)
        value = float(out["value"][i].item())
        if seats[i] != p.root_seat:
            value = -value
        results.append(
            LeafPacket(
                obs=p.obs,
                your_deck=p.your_deck,
                root_seat=p.root_seat,
                value=value,
                priors=priors,
                combos=combos,
            )
        )
    return results


class BatchedLeafServer:
    """Accumulate leaf packets and flush when full or timeout elapses.

    Synchronous v1: callers ``submit`` then ``flush`` / ``flush_if_ready``.
    Async queue + timeout is for a future multi-process infer server; the
    timeout knob is honored here for micro-batching across interleaved trees.
    """

    def __init__(
        self,
        model: TemporalCabtTransformer,
        *,
        batch_size: Optional[int] = None,
        timeout_ms: Optional[float] = None,
    ):
        self.model = model
        self.batch_size = int(
            batch_size if batch_size is not None else config.SEARCH.leaf_batch_size
        )
        self.timeout_ms = float(
            timeout_ms
            if timeout_ms is not None
            else config.SEARCH.inference_batch_timeout_ms
        )
        self._pending: list[LeafPacket] = []
        self._first_submit_t: Optional[float] = None
        self.model.eval()

    def __len__(self) -> int:
        return len(self._pending)

    def submit(self, packet: LeafPacket) -> None:
        if self._first_submit_t is None:
            self._first_submit_t = time.perf_counter()
        self._pending.append(packet)

    def ready(self) -> bool:
        if not self._pending:
            return False
        if len(self._pending) >= self.batch_size:
            return True
        if self._first_submit_t is None:
            return False
        elapsed_ms = (time.perf_counter() - self._first_submit_t) * 1000.0
        return elapsed_ms >= self.timeout_ms and len(self._pending) > 0

    def flush(self) -> list[LeafPacket]:
        if not self._pending:
            return []
        batch = self._pending
        self._pending = []
        self._first_submit_t = None
        return forward_leaf_batch(self.model, batch)

    def flush_if_ready(self) -> list[LeafPacket]:
        if self.ready():
            return self.flush()
        return []

    def evaluate_now(self, packets: Sequence[LeafPacket]) -> list[LeafPacket]:
        """Immediate batch (bypass queue). Chunks by ``batch_size``."""
        out: list[LeafPacket] = []
        bs = max(1, self.batch_size)
        for i in range(0, len(packets), bs):
            out.extend(forward_leaf_batch(self.model, packets[i : i + bs]))
        return out
