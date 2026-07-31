"""GPU-batched policy/value (leaf) inference for MCTS / self-play collect.

CABT ``search_begin`` / ``search_step`` stay per-process and per-tree on CPU.
This module only batches **network** evals: pack board + option SparseVectors,
one ``TemporalCabtTransformer.forward``, scatter values/priors.

See ``outputs/notes/gpu_batched_selfplay.md``.
"""

from __future__ import annotations

import gc
import math
import os
import queue as _queue
import statistics
import time
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from . import cg_env, config, features
from .matchup_adapters import UNKNOWN_ROUTE
from .model import TemporalCabtTransformer


def _mps_cache_release_interval() -> int:
    """Return how often a long-lived MPS leaf releases its allocator cache.

    MPSGraph specializes operations for the input shape.  Self-play changes
    history length and legal-option count almost every move, so a persistent
    leaf otherwise retains a large high-water allocation while it encounters
    those shapes.  Releasing after every batch is the conservative default for
    the shared-memory Macs.  ``0`` remains an explicit performance opt-out.
    """

    raw = os.environ.get("POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES", "1")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1


def _release_mps_cache(*, collect: bool = False) -> None:
    """Best-effort release of memory owned by a long-lived MPS leaf.

    Synchronization must precede ``empty_cache`` so allocations still used by
    queued kernels are eligible for release.  Cleanup must never turn a
    successful inference or checkpoint reload into a server failure.
    """

    if collect:
        gc.collect()
    try:
        torch.mps.synchronize()
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


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
    #: Acting-seat realized board history. Trusted policy requests populate it;
    #: experimental search leaves omit it and are never trusted targets.
    history_boards: Optional[list["features.SparseVector"]] = None
    history_previous_actions: Optional[
        list[Optional["features.SparseVector"]]
    ] = None
    action_combos_override: Optional[list[list[int]]] = None
    value: float = 0.0
    priors: list[float] = field(default_factory=list)
    combos: list[list[int]] = field(default_factory=list)
    # Public-prefix route candidate. The checkpoint-side adapter flag is the
    # final gate; dormant checkpoints consume this as an exact no-op.
    matchup_route: int = UNKNOWN_ROUTE


@dataclass
class FeaturizedLeaves:
    """CPU-side featurization of a batch of leaves (picklable → sendable to the
    GPU inference server). ``combos`` is kept for local child expansion and is
    NOT required by the GPU forward (only ``n_opts`` matters for padding)."""

    boards: list["features.SparseVector"]
    opts: list["features.SparseVector"]
    combos: list[list[list[int]]]
    n_opts: list[int]
    seats: list[int]
    root_seats: list[int]
    histories: Optional[list[list["features.SparseVector"]]] = None
    previous_action_histories: Optional[
        list[list[Optional["features.SparseVector"]]]
    ] = None
    matchup_routes: Optional[list[int]] = None


def featurize_packets(packets: Sequence[LeafPacket]) -> FeaturizedLeaves:
    """Pure-CPU featurization (board + option SparseVectors, combos, seats).

    This is the part that stays distributed across the CPU sim workers; only the
    resulting tensors are shipped to the single GPU server for the forward pass.
    Info-set integrity is enforced per leaf here.
    """
    boards: list = []
    opts: list = []
    combos_list: list[list[list[int]]] = []
    n_opts: list[int] = []
    seats: list[int] = []
    root_seats: list[int] = []
    matchup_routes: list[int] = []
    histories: list[list["features.SparseVector"]] = []
    previous_action_histories: list[list[Optional["features.SparseVector"]]] = []
    for p in packets:
        features.assert_info_set(p.obs)
        board = features.build_board_tokens(p.obs, p.your_deck)
        boards.append(board)
        histories.append(list(p.history_boards) if p.history_boards else [board])
        previous_action_histories.append(
            list(p.history_previous_actions)
            if p.history_previous_actions is not None
            else [None] * len(histories[-1])
        )
        combos = (
            [list(combo) for combo in p.action_combos_override]
            if p.action_combos_override is not None
            else features.enumerate_action_combos(p.obs)
        )
        opts.append(features.build_option_tokens(p.obs, combos))
        combos_list.append(combos)
        n_opts.append(len(combos))
        obs_obj = cg_env.to_observation(p.obs) if isinstance(p.obs, dict) else p.obs
        seats.append(
            obs_obj.current.yourIndex if obs_obj.current is not None else p.root_seat
        )
        root_seats.append(p.root_seat)
        route = p.matchup_route
        if type(route) is not int or route < UNKNOWN_ROUTE:
            raise ValueError("leaf matchup route must be an exact route integer")
        matchup_routes.append(route)
    return FeaturizedLeaves(
        boards,
        opts,
        combos_list,
        n_opts,
        seats,
        root_seats,
        histories,
        previous_action_histories,
        matchup_routes,
    )


@torch.no_grad()
def _forward_chunk(
    model: TemporalCabtTransformer,
    boards: list,
    opts: list,
    n_opts: list,
    seats: list,
    root_seats: list,
    autocast_dtype: Optional[torch.dtype],
    histories: Optional[list[list["features.SparseVector"]]] = None,
    previous_action_histories: Optional[
        list[list[Optional["features.SparseVector"]]]
    ] = None,
    matchup_routes: Optional[list[int]] = None,
) -> list[tuple[float, list[float]]]:
    dev = next(model.parameters()).device
    use_ac = autocast_dtype is not None and dev.type in {"cuda", "mps"}
    ctx = (
        torch.autocast(dev.type, dtype=autocast_dtype)
        if use_ac
        else _nullcontext()
    )
    with ctx:
        if model.decision_context == "history":
            out = model.forward_history_batch(
                histories or [[board] for board in boards],
                opts,
                n_options=n_opts,
                previous_action_histories=previous_action_histories,
                matchup_routes=(
                    matchup_routes
                    if matchup_routes is not None
                    else [UNKNOWN_ROUTE] * len(boards)
                ),
            )
        else:
            out = model.forward(
                boards,
                opts,
                kv_cache=None,
                append_cache=False,
                n_options=n_opts,
                matchup_routes=(
                    matchup_routes
                    if matchup_routes is not None
                    else [UNKNOWN_ROUTE] * len(boards)
                ),
            )
    logits_all = out["policy_logits"]
    value_all = out["value"]
    # Copy each result tensor to the host once.  Calling ``.cpu()`` / ``.item()``
    # per row serializes MPS command buffers and turns a unified-memory batch
    # into 2*B host/device synchronization points.  A single bulk readback is
    # numerically identical and also reduces CUDA launch synchronization.
    logits_host = logits_all.detach().float().cpu()
    values_host = value_all.detach().float().cpu()
    results: list[tuple[float, list[float]]] = []
    for i in range(len(boards)):
        n = int(n_opts[i])
        priors = _policy_probs(logits_host[i, :n])
        value = float(values_host[i].item())
        if seats[i] != root_seats[i]:
            value = -value
        results.append((value, priors))
    return results


@torch.no_grad()
def forward_featurized(
    model: TemporalCabtTransformer,
    boards: Sequence,
    opts: Sequence,
    n_opts: Sequence[int],
    seats: Sequence[int],
    root_seats: Sequence[int],
    *,
    autocast_dtype: Optional[torch.dtype] = None,
    histories: Optional[Sequence[Sequence["features.SparseVector"]]] = None,
    previous_action_histories: Optional[
        Sequence[Sequence[Optional["features.SparseVector"]]]
    ] = None,
    matchup_routes: Optional[Sequence[int]] = None,
) -> list[tuple[float, list[float]]]:
    """GPU forward over pre-featurized leaves → ``[(value, priors), ...]``.

    Value is flipped to each leaf's ``root_seat`` perspective. This is the ONLY
    part that must run on the GPU; it holds no per-tree / CABT state.

    VRAM-safe: on a CUDA OOM (a too-large coalesced batch on a busy card), free
    the cache and recursively split the batch in half until it fits — never
    crash the server. Progress is guaranteed (down to a single leaf).
    """
    b = list(boards)
    if not b:
        return []
    o, no, se, rs = list(opts), list(n_opts), list(seats), list(root_seats)
    mr = list(matchup_routes) if matchup_routes is not None else None
    if mr is not None and len(mr) != len(b):
        raise ValueError("leaf matchup route count does not match batch size")
    try:
        hist = [list(h) for h in histories] if histories is not None else None
        return _forward_chunk(
            model,
            b,
            o,
            no,
            se,
            rs,
            autocast_dtype,
            histories=hist,
            previous_action_histories=(
                [list(h) for h in previous_action_histories]
                if previous_action_histories is not None
                else None
            ),
            matchup_routes=mr,
        )
    except Exception as exc:  # noqa: BLE001
        if not config.is_cuda_oom(exc) or len(b) <= 1:
            raise
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        mid = len(b) // 2
        left = forward_featurized(
            model, b[:mid], o[:mid], no[:mid], se[:mid], rs[:mid],
            autocast_dtype=autocast_dtype,
            histories=(list(histories)[:mid] if histories is not None else None),
            previous_action_histories=(
                list(previous_action_histories)[:mid]
                if previous_action_histories is not None
                else None
            ),
            matchup_routes=(mr[:mid] if mr is not None else None),
        )
        right = forward_featurized(
            model, b[mid:], o[mid:], no[mid:], se[mid:], rs[mid:],
            autocast_dtype=autocast_dtype,
            histories=(list(histories)[mid:] if histories is not None else None),
            previous_action_histories=(
                list(previous_action_histories)[mid:]
                if previous_action_histories is not None
                else None
            ),
            matchup_routes=(mr[mid:] if mr is not None else None),
        )
        return left + right


@torch.no_grad()
def forward_leaf_batch(
    model: TemporalCabtTransformer,
    packets: Sequence[LeafPacket],
) -> list[LeafPacket]:
    """Vectorized policy/value over ``packets`` (pad options, mask illegal).

    Returns new :class:`LeafPacket` instances with ``value`` / ``priors`` /
    ``combos`` filled (value in each packet's ``root_seat`` perspective). Local
    (single-process) path — featurize + forward on ``model``'s own device.
    """
    if not packets:
        return []
    fl = featurize_packets(packets)
    vp = forward_featurized(
        model,
        fl.boards,
        fl.opts,
        fl.n_opts,
        fl.seats,
        fl.root_seats,
        histories=fl.histories,
        previous_action_histories=fl.previous_action_histories,
        matchup_routes=fl.matchup_routes,
    )
    results: list[LeafPacket] = []
    for i, p in enumerate(packets):
        results.append(
            LeafPacket(
                obs=p.obs,
                your_deck=p.your_deck,
                root_seat=p.root_seat,
                history_boards=p.history_boards,
                history_previous_actions=p.history_previous_actions,
                action_combos_override=p.action_combos_override,
                value=vp[i][0],
                priors=vp[i][1],
                combos=fl.combos[i],
                matchup_route=p.matchup_route,
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
        if batch_size is None:
            # Size the GPU forward width to the card the model lives on
            # (512 Blackwell / 256 3080 Ti) so batched inference saturates it.
            try:
                dev = next(model.parameters()).device
                batch_size = config.leaf_batch_for_device(dev)
            except Exception:
                batch_size = config.SEARCH.leaf_batch_size
        self.batch_size = int(batch_size)
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


# ---------------------------------------------------------------------------
# Persistent GPU-resident batched leaf-eval server (decoupled from CPU workers)
# ---------------------------------------------------------------------------
#
# Architecture (fixes idle-GPU during the round-robin GAME phase):
#   * CPU sim/baseline workers stay CUDA_VISIBLE_DEVICES="" (no per-worker CUDA
#     context → no OOM). They FEATURIZE leaves (parallel, cheap) and ship the
#     board/option SparseVectors to this one GPU server via a shared request
#     queue, then block on their own response queue.
#   * ONE server process keeps the champion weights resident on the GPU. It
#     COALESCES leaves across ALL concurrent games into one forward → fills the
#     GPU during self-play. Weights are hot-reloaded after each promotion.
#
# Request  (worker → server): (slot:int, rid:int, FeaturizedLeaves)
# Response (server → worker): (rid:int, [(value, priors), ...])  on resp_qs[slot]
# Control  (parent → server): ("reload", ckpt_path) | ("stop", None)

# Worker-local handle to the server, populated by the pool initializer.
_REMOTE: dict = {}


class RemoteLeafError(RuntimeError):
    """Remote inference failed or returned an invalid response."""


class RemoteLeafTimeout(RemoteLeafError, TimeoutError):
    """Remote inference exceeded its bounded request deadline."""


class RemoteLeafCancelled(RemoteLeafError):
    """The owning simulator pool cancelled this request generation."""


def set_remote_leaf_channel(
    slot: int,
    req_q,
    resp_q,
    *,
    generation: int = 0,
    alive_evt=None,
    expected_digest: object = None,
    expected_version: object = None,
    timeout_s: Optional[float] = None,
    stop_event=None,
    req_qs=None,
    leaf_devices=None,
    alive_evts=None,
    home_server_idx: Optional[int] = None,
) -> None:
    """Register this worker's slot + queues (called once in the pool init)."""
    _REMOTE.clear()
    _REMOTE["slot"] = int(slot)
    _REMOTE["req_q"] = req_q
    _REMOTE["resp_q"] = resp_q
    _REMOTE["generation"] = int(generation)
    _REMOTE["alive_evt"] = alive_evt
    _REMOTE["expected_digest"] = expected_digest
    _REMOTE["expected_version"] = expected_version
    _REMOTE["stop_event"] = stop_event
    _REMOTE["timeout_s"] = float(
        timeout_s
        if timeout_s is not None
        else config.SEARCH.remote_request_timeout_s
    )
    # Optional multi-server fan-out: least-queue pick at request time so idle
    # GPU0 leaves steal work instead of waiting on sticky home-server waves.
    _REMOTE["req_qs"] = list(req_qs) if req_qs is not None else None
    _REMOTE["leaf_devices"] = list(leaf_devices) if leaf_devices is not None else None
    _REMOTE["alive_evts"] = list(alive_evts) if alive_evts is not None else None
    _REMOTE["home_server_idx"] = (
        int(home_server_idx) if home_server_idx is not None else None
    )


def has_remote_leaf_channel() -> bool:
    return "req_q" in _REMOTE


class RemoteLeafClient:
    """Leaf backend that offloads the GPU forward to the persistent server.

    Callable ``(packets) -> list[LeafPacket]``: featurize on this CPU worker,
    send tensors to the server, block for the (value, priors) results, and
    rebuild packets with the locally-known combos. One in-flight request per
    worker (the game loop is synchronous), so ``rid`` just guards ordering.
    """

    def __init__(
        self,
        slot: int,
        req_q,
        resp_q,
        *,
        generation: int = 0,
        alive_evt=None,
        expected_digest: object = None,
        expected_version: object = None,
        timeout_s: Optional[float] = None,
        stop_event=None,
        req_qs=None,
        leaf_devices=None,
        alive_evts=None,
        home_server_idx: Optional[int] = None,
    ) -> None:
        self.slot = int(slot)
        self.req_q = req_q
        self.resp_q = resp_q
        self.generation = int(generation)
        self.alive_evt = alive_evt
        self.expected_digest = expected_digest
        self.expected_version = expected_version
        self.stop_event = stop_event
        self.req_qs = list(req_qs) if req_qs is not None else None
        self.leaf_devices = list(leaf_devices) if leaf_devices is not None else None
        self.alive_evts = list(alive_evts) if alive_evts is not None else None
        self.home_server_idx = (
            int(home_server_idx) if home_server_idx is not None else None
        )
        self._deadline_monotonic: Optional[float] = None
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else config.SEARCH.remote_request_timeout_s
        )
        self._rid = 0
        self._telemetry_events: list[dict[str, float]] = []

    @staticmethod
    def _expected_field(source: object, field: str) -> object:
        """Read a scalar expectation or a live manager-dict field.

        Remote workers intentionally receive the same ``Manager().dict()`` for
        digest and version so a checkpoint reload is visible without respawning
        the simulator pool.  Resolve that proxy for every response instead of
        comparing the response to the proxy object itself.
        """
        getter = getattr(source, "get", None)
        if callable(getter):
            return getter(field)
        return source

    def _select_req_target(self):
        """Home sticky queue, or least-queue across replicas (GPU0-preferring)."""
        if not self.req_qs or len(self.req_qs) <= 1:
            return self.req_q, self.alive_evt
        from poke_bot.pure_rl.hardware import pick_leaf_server_index

        devices = self.leaf_devices or [1] * len(self.req_qs)
        idx = pick_leaf_server_index(
            req_qs=self.req_qs,
            devices=devices,
            alive_evts=self.alive_evts,
            preferred_index=self.home_server_idx,
        )
        alive = None
        if self.alive_evts is not None and idx < len(self.alive_evts):
            alive = self.alive_evts[idx]
        elif idx == 0:
            alive = self.alive_evt
        return self.req_qs[idx], alive

    def set_deadline(self, deadline_monotonic: Optional[float]) -> None:
        """Bound subsequent RPC waits by the active search move deadline."""
        self._deadline_monotonic = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else None
        )

    def _remaining_timeout(self) -> float:
        timeout = self.timeout_s
        if self._deadline_monotonic is not None:
            timeout = min(
                timeout,
                max(0.0, self._deadline_monotonic - time.monotonic()),
            )
        return timeout

    def _raise_if_cancelled(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise RemoteLeafCancelled(
                f"leaf generation {self.generation} was cancelled"
            )

    def telemetry_mark(self) -> int:
        """Return a marker for per-move remote-inference telemetry."""
        return len(self._telemetry_events)

    def telemetry_since(self, marker: int) -> dict[str, float | int]:
        """Summarize server queue/batch timing since ``marker``."""
        events = self._telemetry_events[max(0, int(marker)) :]
        if not events:
            return {
                "remote_requests": 0,
                "remote_leaves": 0,
                "queue_wait_ms_mean": 0.0,
                "queue_wait_ms_p95": 0.0,
                "inference_batch_size_mean": 0.0,
                "inference_batch_size_p95": 0.0,
                "server_inference_ms_mean": 0.0,
                "client_roundtrip_ms_mean": 0.0,
                "request_queue_depth_mean": 0.0,
                "request_queue_depth_max": 0,
                "server_queue_depth_mean": 0.0,
                "batch_occupancy_mean": 0.0,
                "coalesced_requests_mean": 0.0,
                "server_request_rate": 0.0,
            }

        def percentile(values: list[float], q: float) -> float:
            ordered = sorted(values)
            index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
            return float(ordered[index])

        queue_ms = [row["queue_wait_ms"] for row in events]
        batch = [row["batch_size"] for row in events]
        inference_ms = [row["inference_ms"] for row in events]
        roundtrip_ms = [row["roundtrip_ms"] for row in events]
        request_depth = [row["request_queue_depth"] for row in events]
        server_depth = [row["server_queue_depth"] for row in events]
        return {
            "remote_requests": len(events),
            "remote_leaves": int(sum(row["request_leaves"] for row in events)),
            "queue_wait_ms_mean": statistics.fmean(queue_ms),
            "queue_wait_ms_p95": percentile(queue_ms, 0.95),
            "inference_batch_size_mean": statistics.fmean(batch),
            "inference_batch_size_p95": percentile(batch, 0.95),
            "server_inference_ms_mean": statistics.fmean(inference_ms),
            "client_roundtrip_ms_mean": statistics.fmean(roundtrip_ms),
            "request_queue_depth_mean": statistics.fmean(request_depth),
            "request_queue_depth_max": int(max(request_depth)),
            "server_queue_depth_mean": statistics.fmean(server_depth),
            "batch_occupancy_mean": statistics.fmean(
                row["batch_occupancy"] for row in events
            ),
            "coalesced_requests_mean": statistics.fmean(
                row["coalesced_requests"] for row in events
            ),
            "server_request_rate": events[-1]["server_request_rate"],
        }

    def __call__(self, packets: Sequence[LeafPacket]) -> list[LeafPacket]:
        if not packets:
            return []
        self._raise_if_cancelled()
        fl = featurize_packets(packets)
        self._rid += 1
        rid = self._rid
        req_q, alive_evt = self._select_req_target()
        put = getattr(req_q, "put", None)
        if not callable(put):
            raise TypeError(
                f"RemoteLeafClient.req_q has no put() (got {type(req_q)!r}); "
                "worker_pool likely passed a list of queues as the request queue "
                "instead of one Queue (multi-replica shard bug)."
            )
        if alive_evt is not None and not alive_evt.is_set():
            raise RemoteLeafError("leaf server is not alive")
        try:
            request_queue_depth = max(0, int(req_q.qsize()))
        except (AttributeError, NotImplementedError, OSError):
            request_queue_depth = 0
        request = {
            "slot": self.slot,
            "generation": self.generation,
            "rid": rid,
            "leaves": fl,
            "enqueued_ns": time.monotonic_ns(),
            "client_queue_depth": request_queue_depth,
        }
        request_t0 = time.perf_counter()
        request_timeout = self._remaining_timeout()
        if request_timeout <= 0:
            raise RemoteLeafTimeout("leaf request exceeded the move deadline")
        try:
            req_q.put(request, timeout=request_timeout)
        except _queue.Full as exc:
            raise RemoteLeafTimeout("leaf request queue put timed out") from exc
        deadline = time.monotonic() + request_timeout
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteLeafTimeout(
                    f"leaf response timed out after {request_timeout:.3f}s "
                    f"(generation={self.generation} rid={rid})"
                )
            if alive_evt is not None and not alive_evt.is_set():
                raise RemoteLeafError("leaf server died while request was in flight")
            try:
                response = self.resp_q.get(timeout=min(remaining, 0.5))
            except _queue.Empty:
                continue
            if not isinstance(response, dict):
                raise RemoteLeafError(
                    f"invalid leaf response type: {type(response).__name__}"
                )
            if (
                int(response.get("generation", -1)) != self.generation
                or int(response.get("rid", -1)) != rid
            ):
                # Stale response from a previous pool generation. Never route it
                # to the current request.
                continue
            if not response.get("ok", False):
                raise RemoteLeafError(str(response.get("error") or "leaf server error"))
            expected_digest = self._expected_field(
                self.expected_digest, "digest"
            )
            expected_version = self._expected_field(
                self.expected_version, "version"
            )
            if (
                expected_digest is not None
                and response.get("checkpoint_digest") != expected_digest
            ):
                raise RemoteLeafError(
                    "leaf response checkpoint digest mismatch: expected "
                    f"{expected_digest}, got {response.get('checkpoint_digest')}"
                )
            if (
                expected_version is not None
                and int(response.get("version", -1)) != int(expected_version)
            ):
                raise RemoteLeafError(
                    "leaf response checkpoint version mismatch: expected "
                    f"{expected_version}, got {response.get('version')}"
                )
            vp = response.get("values")
            if not isinstance(vp, list) or len(vp) != len(packets):
                raise RemoteLeafError(
                    f"leaf response length mismatch: expected {len(packets)}, "
                    f"got {len(vp) if isinstance(vp, list) else type(vp).__name__}"
                )
            break
        event = {
            "queue_wait_ms": float(response.get("queue_wait_ms", 0.0)),
            "batch_size": float(response.get("batch_size", len(packets))),
            "inference_ms": float(response.get("inference_ms", 0.0)),
            "roundtrip_ms": (time.perf_counter() - request_t0) * 1000.0,
            "request_leaves": float(len(packets)),
            "request_queue_depth": float(request_queue_depth),
            "server_queue_depth": float(
                response.get("server_queue_depth", 0.0)
            ),
            "batch_occupancy": float(response.get("batch_occupancy", 0.0)),
            "coalesced_requests": float(
                response.get("coalesced_requests", 1.0)
            ),
            "server_request_rate": float(
                response.get("server_request_rate", 0.0)
            ),
        }
        self._telemetry_events.append(event)
        # One search move is bounded well below this. Keep long-running workers
        # memory-bounded while preserving enough history for p95 summaries.
        if len(self._telemetry_events) > 8192:
            del self._telemetry_events[:4096]
        return [
            LeafPacket(
                obs=p.obs,
                your_deck=p.your_deck,
                root_seat=p.root_seat,
                history_boards=p.history_boards,
                history_previous_actions=p.history_previous_actions,
                action_combos_override=p.action_combos_override,
                value=vp[i][0],
                priors=vp[i][1],
                combos=fl.combos[i],
                matchup_route=p.matchup_route,
            )
            for i, p in enumerate(packets)
        ]


def remote_leaf_backend_from_worker():
    """Build a :class:`RemoteLeafClient` from this worker's registered channel,
    or ``None`` if no server is attached (fall back to local eval)."""
    if not has_remote_leaf_channel():
        return None
    return RemoteLeafClient(
        _REMOTE["slot"],
        _REMOTE["req_q"],
        _REMOTE["resp_q"],
        generation=_REMOTE.get("generation", 0),
        alive_evt=_REMOTE.get("alive_evt"),
        expected_digest=_REMOTE.get("expected_digest"),
        expected_version=_REMOTE.get("expected_version"),
        timeout_s=_REMOTE.get("timeout_s"),
        stop_event=_REMOTE.get("stop_event"),
        req_qs=_REMOTE.get("req_qs"),
        leaf_devices=_REMOTE.get("leaf_devices"),
        alive_evts=_REMOTE.get("alive_evts"),
        home_server_idx=_REMOTE.get("home_server_idx"),
    )


def apply_runtime_matchup_adapter_contract(model) -> bool:
    """Enable the activated adapter bank on any freshly loaded leaf model."""

    if os.environ.get("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "").strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return False
    from .public_matchup_router import load_runtime_public_matchup_tree

    tree_path = os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "").strip()
    if not tree_path:
        raise ValueError(
            "runtime matchup adapters require POKEBOT_PUBLIC_MATCHUP_TREE_PATH"
        )
    load_runtime_public_matchup_tree(tree_path)
    model.matchup_adapter_bank.enabled = True
    model.matchup_adapter_bank.requires_grad_(False)
    return True


def run_leaf_server(
    ckpt_path: str,
    device_str: str,
    req_q,
    resp_qs: list,
    *,
    ready_evt=None,
    alive_evt=None,
    ctrl_q=None,
    status_q=None,
    expected_digest: Optional[str] = None,
    initial_version: int = 0,
    max_batch: Optional[int] = None,
    coalesce_ms: float = 2.0,
    bf16: bool = True,
) -> None:
    """Persistent GPU inference server loop (run in its own spawned process).

    Keeps the champion resident on ``device_str``; coalesces leaves across all
    worker requests into one forward; scatters (value, priors) back per request.
    """
    # This process is the ONLY one allowed a CUDA context for our net; the
    # parent sets workers to CUDA_VISIBLE_DEVICES="" but the server must see the
    # GPU, so we do NOT clear it here.
    from .checkpoint import checkpoint_digest
    from .train import load_model_from_checkpoint

    def _status(payload: dict) -> None:
        if status_q is not None:
            status_q.put(payload)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device_str)
    version = int(initial_version)
    current_digest = ""

    try:
        current_digest = checkpoint_digest(ckpt_path)
        if expected_digest is not None and current_digest != expected_digest:
            raise ValueError(
                f"initial checkpoint digest mismatch: expected {expected_digest}, "
                f"got {current_digest}"
            )
        model = load_model_from_checkpoint(ckpt_path, device=device)
        apply_runtime_matchup_adapter_contract(model)
        model.eval()
    except BaseException as exc:  # noqa: BLE001 - report startup failure to parent
        _status(
            {
                "type": "ready",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "version": version,
                "checkpoint_digest": current_digest,
            }
        )
        if ready_evt is not None:
            ready_evt.set()
        return

    ac_dtype = torch.bfloat16 if (bf16 and device.type == "cuda") else None
    if device.type == "mps":
        mps_autocast = os.environ.get(
            "POKEBOT_MPS_AUTOCAST_DTYPE", "float32"
        ).strip().lower()
        if mps_autocast in {"bf16", "bfloat16"}:
            ac_dtype = torch.bfloat16
        elif mps_autocast in {"fp16", "float16"}:
            ac_dtype = torch.float16
        elif mps_autocast not in {"", "none", "off", "fp32", "float32"}:
            raise ValueError(
                "POKEBOT_MPS_AUTOCAST_DTYPE must be float32, float16, or bfloat16"
            )
    if max_batch is None:
        try:
            max_batch = int(config.leaf_batch_for_device(device))
        except Exception:
            max_batch = 256
    max_batch = max(1, int(max_batch))
    coalesce_s = max(0.0, float(coalesce_ms) / 1000.0)

    if alive_evt is not None:
        alive_evt.set()
    _status(
        {
            "type": "ready",
            "ok": True,
            "version": version,
            "checkpoint_digest": current_digest,
        }
    )
    if ready_evt is not None:
        ready_evt.set()

    stop = False
    cancelled_generations: set[int] = set()
    server_started = time.perf_counter()
    requests_served = 0
    inference_batches = 0
    mps_cache_every = (
        _mps_cache_release_interval() if device.type == "mps" else 0
    )
    try:
        while not stop:
            # Drain control messages before accepting another inference batch.
            if ctrl_q is not None:
                try:
                    while True:
                        msg = ctrl_q.get_nowait()
                        if isinstance(msg, dict):
                            cmd = msg.get("cmd")
                            arg = msg
                        else:
                            cmd, legacy_arg = msg
                            arg = {"path": legacy_arg}
                        if cmd == "reload":
                            requested_version = int(arg.get("version", version + 1))
                            requested_digest = arg.get("digest")
                            candidate_model = None
                            try:
                                path = str(arg["path"])
                                actual = checkpoint_digest(path)
                                if (
                                    requested_digest is not None
                                    and actual != requested_digest
                                ):
                                    raise ValueError(
                                        f"reload digest mismatch: expected "
                                        f"{requested_digest}, got {actual}"
                                    )
                                candidate_model = load_model_from_checkpoint(
                                    path, device=device
                                )
                                apply_runtime_matchup_adapter_contract(candidate_model)
                                candidate_model.eval()
                                old_model = model
                                model = candidate_model
                                candidate_model = None
                                del old_model
                                if device.type == "mps":
                                    _release_mps_cache(collect=True)
                                current_digest = actual
                                version = requested_version
                                _status(
                                    {
                                        "type": "reload",
                                        "ok": True,
                                        "version": version,
                                        "checkpoint_digest": current_digest,
                                    }
                                )
                            except BaseException as exc:  # noqa: BLE001
                                candidate_model = None
                                if device.type == "mps":
                                    _release_mps_cache(collect=True)
                                _status(
                                    {
                                        "type": "reload",
                                        "ok": False,
                                        "version": requested_version,
                                        "checkpoint_digest": current_digest,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                        elif cmd == "stop":
                            stop = True
                            break
                        elif cmd == "cancel_generation":
                            cancelled_generations.add(
                                int(arg.get("generation", -1))
                            )
                except _queue.Empty:
                    pass
            if stop:
                break

            try:
                first = req_q.get(timeout=0.25)
            except _queue.Empty:
                continue
            if first is None:
                break
            if not isinstance(first, dict):
                continue
            if int(first.get("generation", -1)) in cancelled_generations:
                continue

            reqs = [first]
            first_fl = first.get("leaves")
            n_leaves = len(first_fl.boards) if first_fl is not None else 0
            if coalesce_s > 0 and n_leaves < max_batch:
                time.sleep(coalesce_s)
            while n_leaves < max_batch:
                try:
                    nxt = req_q.get_nowait()
                except _queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                if not isinstance(nxt, dict):
                    continue
                if int(nxt.get("generation", -1)) in cancelled_generations:
                    continue
                reqs.append(nxt)
                fl = nxt.get("leaves")
                n_leaves += len(fl.boards) if fl is not None else 0

            boards: list = []
            opts: list = []
            n_opts: list[int] = []
            seats: list[int] = []
            root_seats: list[int] = []
            matchup_routes: list[int] = []
            histories: list[list["features.SparseVector"]] = []
            previous_action_histories: list[
                list[Optional["features.SparseVector"]]
            ] = []
            valid_reqs: list[tuple[dict, FeaturizedLeaves]] = []
            for req in reqs:
                fl = req.get("leaves")
                slot = int(req.get("slot", -1))
                if not isinstance(fl, FeaturizedLeaves) or not 0 <= slot < len(resp_qs):
                    continue
                valid_reqs.append((req, fl))
                boards.extend(fl.boards)
                opts.extend(fl.opts)
                n_opts.extend(fl.n_opts)
                seats.extend(fl.seats)
                root_seats.extend(fl.root_seats)
                matchup_routes.extend(
                    fl.matchup_routes
                    if fl.matchup_routes is not None
                    else [UNKNOWN_ROUTE] * len(fl.boards)
                )
                histories.extend(fl.histories or [[board] for board in fl.boards])
                previous_action_histories.extend(
                    fl.previous_action_histories
                    or [
                        [None] * len(history)
                        for history in (
                            fl.histories or [[board] for board in fl.boards]
                        )
                    ]
                )

            batch_started_ns = time.monotonic_ns()
            try:
                server_queue_depth = max(0, int(req_q.qsize()))
            except (AttributeError, NotImplementedError, OSError):
                server_queue_depth = 0
            inference_t0 = time.perf_counter()
            try:
                vp = forward_featurized(
                    model,
                    boards,
                    opts,
                    n_opts,
                    seats,
                    root_seats,
                    autocast_dtype=ac_dtype,
                    histories=histories,
                    previous_action_histories=previous_action_histories,
                    matchup_routes=matchup_routes,
                )
                error = None
            except BaseException as exc:  # noqa: BLE001 - route failure to every caller
                vp = []
                error = f"{type(exc).__name__}: {exc}"
            finally:
                inference_batches += 1
                if (
                    mps_cache_every > 0
                    and inference_batches % mps_cache_every == 0
                ):
                    _release_mps_cache()
            inference_ms = (time.perf_counter() - inference_t0) * 1000.0
            requests_served += len(valid_reqs)
            request_rate = requests_served / max(
                time.perf_counter() - server_started, 1e-9
            )

            off = 0
            for req, fl in valid_reqs:
                slot = int(req["slot"])
                k = len(fl.boards)
                response = {
                    "generation": int(req.get("generation", 0)),
                    "rid": int(req.get("rid", -1)),
                    "ok": error is None,
                    "values": vp[off : off + k] if error is None else None,
                    "error": error,
                    "version": version,
                    "checkpoint_digest": current_digest,
                    "queue_wait_ms": max(
                        0.0,
                        (
                            batch_started_ns
                            - int(req.get("enqueued_ns", batch_started_ns))
                        )
                        / 1e6,
                    ),
                    "batch_size": len(boards),
                    "coalesced_requests": len(valid_reqs),
                    "server_queue_depth": server_queue_depth,
                    "batch_occupancy": len(boards) / max_batch,
                    "server_request_rate": request_rate,
                    "inference_ms": inference_ms,
                }
                try:
                    # A cancelled/dead worker must never wedge the persistent
                    # server by leaving its bounded response queue full.
                    resp_qs[slot].put(response, timeout=0.1)
                except _queue.Full:
                    pass
                off += k
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        if device.type == "mps":
            _release_mps_cache(collect=True)
        if alive_evt is not None:
            alive_evt.clear()
