"""Project configuration: the single SOURCE OF TRUTH for the hardware profile.

Values are read once at import from module-level defaults, each overridable via
an environment variable (``POKEBOT_<NAME>``) so scripts can tune without code
edits. Every system (dataset/train/core_kernel/self_play/mcts/collect/eval)
imports these constants + the helper API below so they all pull the *same*
aggressive "full hardware" knobs.

BOLD / dedicated-box policy
---------------------------
This machine is DEDICATED solely to this training — the profile intentionally
leaves **no headroom**:
  CPU   Ryzen 7950X (16C / 32T)   -> use ALL 32 threads for sim/feature/dataloader
  RAM   128 GB (+256 GB swap)     -> big RAM-resident tensor/replay caches
  GPU0  RTX 3080 Ti (12 GB)       -> core-kernel/specialist train + batched leaf eval
  GPU1  RTX PRO 5000 Blackwell 48G -> primary archetype train / RL / batched inference

Nothing here changes model math, info-set safety, or resumability — only
parallelism / throughput / utilisation defaults. Everything is still env-
overridable for the odd shared-box case.

Helper API (import once, used everywhere)
-----------------------------------------
  * :func:`gpu_kind`            — classify a torch device → ``"blackwell"`` /
    ``"3080ti"`` / ``"cpu"`` (PCI-order-safe; reconciles with ``device.py``).
  * :func:`batch_profile`       — per-GPU :class:`BatchProfile` (batch sizes,
    shuffle buffer, leaf-eval batch, AMP dtype, grad-accum).
  * :func:`dataloader_kwargs`   — ``torch.utils.data.DataLoader`` kwargs
    (num_workers / pin_memory / persistent_workers / prefetch_factor).
  * :func:`leaf_batch_for_device` — GPU-sized MCTS leaf-eval batch size.
  * :func:`apply_runtime_perf`  — set TF32 / cuDNN benchmark / thread counts.
  * :func:`hardware_profile`    — one resolved dict for logging / dry-runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import paths


def _set_default_env() -> None:
    """Set process-wide env defaults that MUST land before torch/CUDA init.

    Uses ``setdefault`` so an explicit operator override always wins. Safe in a
    torch-less environment (only touches ``os.environ``). Imported-early because
    ``config`` is imported by essentially every entry point.
    """
    # Reduce fragmentation + allow the allocator to grow/shrink segments so a
    # big transient activation spike is far less likely to hard-OOM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # PCI bus order so torch indices match nvidia-smi (idx0=3080 Ti, idx1=Blackwell).
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Give math libs all threads by default (torch.set_num_threads refines later).
    _threads = str(os.cpu_count() or 32)
    os.environ.setdefault("OMP_NUM_THREADS", _threads)
    os.environ.setdefault("MKL_NUM_THREADS", _threads)


_set_default_env()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(f"POKEBOT_{name}")
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(f"POKEBOT_{name}")
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(f"POKEBOT_{name}")
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(f"POKEBOT_{name}")
    return Path(v).expanduser() if v else default


# ---------------------------------------------------------------------------
# Card vocabulary (from all_card_data(); the model card-embedding table size).
# Defaults reflect the current engine (max cardId 1267 -> vocab 1268; max
# attackId 1556 -> vocab 1557). Call refresh_vocab() to recompute from the live
# cg library when it is importable.
# ---------------------------------------------------------------------------

CARD_VOCAB: int = _env_int("CARD_VOCAB", 1268)
ATTACK_VOCAB: int = _env_int("ATTACK_VOCAB", 1557)


def agent_verbose() -> bool:
    """Whether baseline / agent ``print()`` debug output is allowed through.

    Default False → round-robin / eval workers redirect agent stdout to
    ``os.devnull`` so the log shows only the tqdm win-rate bar. Set
    ``POKEBOT_AGENT_VERBOSE=1`` (or pass ``--agent-verbose``) to restore the
    full HAND/OPTIONS/etc. baseline debug spam for troubleshooting.
    """
    return _env_bool("AGENT_VERBOSE", False)


def refresh_vocab() -> tuple[int, int]:
    """Recompute (CARD_VOCAB, ATTACK_VOCAB) from the live cg library.

    Mutates the module-level globals and returns the new pair. Safe to call at
    startup once the competition data / cg runtime is present.
    """
    global CARD_VOCAB, ATTACK_VOCAB
    from .cg_env import all_card_data, all_attack

    CARD_VOCAB = max((c.cardId for c in all_card_data()), default=CARD_VOCAB - 1) + 1
    ATTACK_VOCAB = max((a.attackId for a in all_attack()), default=ATTACK_VOCAB - 1) + 1
    return CARD_VOCAB, ATTACK_VOCAB


#: Physical core / thread counts of the dedicated Ryzen 7950X (16C / 32T).
CPU_THREADS: int = os.cpu_count() or 32


@dataclass
class HardwareConfig:
    """CPU / worker / cache defaults — BOLD but crash-safe for unattended runs.

    Dedicated box: use all 32 threads for CPU parallelism, budget RAM caches
    aggressively but strictly inside PHYSICAL RAM (never design for swap), and
    keep per-worker × worker-count memory inside the RAM budget so the Linux
    OOM killer never fires on an unattended run.
    """

    #: libcg sim workers (self-play / eval / round-robin). libcg battles are
    #: lightweight per process → run one per hardware thread (all 32).
    sim_workers: int = _env_int("SIM_WORKERS", min(CPU_THREADS, 32))
    #: Replay featurization / classification / scan workers (all 32 threads).
    feature_workers: int = _env_int("FEATURE_WORKERS", min(CPU_THREADS, 32))
    #: torch ``DataLoader`` workers. Bounded BELOW the thread count on purpose:
    #: each loader worker can hold prefetched/cached tensors, so we cap the
    #: count so ``per_worker_mem × workers`` stays within the RAM budget rather
    #: than oversubscribing and inviting the OOM killer.
    dataloader_workers: int = _env_int("DATALOADER_WORKERS", 16)
    pin_memory: bool = _env_bool("PIN_MEMORY", True)
    #: Keep loader workers alive across epochs (avoids re-spawn/re-import cost).
    persistent_workers: bool = _env_bool("PERSISTENT_WORKERS", True)
    #: Batches each loader worker prefetches ahead (bounded → bounded RAM).
    prefetch_factor: int = _env_int("PREFETCH_FACTOR", 4)

    #: threads handed to torch intra-op pools (``torch.set_num_threads``).
    torch_threads: int = _env_int("TORCH_THREADS", min(CPU_THREADS, 32))

    #: In-process concurrent MCTS trees / games that share one leaf batcher.
    #: libcg battle is still one-at-a-time per process; this sizes multi-tree
    #: search + reanalyse collect (see ``poke_bot.self_play``).
    parallel_games: int = _env_int("PARALLEL_GAMES", 16)

    #: Recycle each sim worker process after this many games. Bounds libcg's
    #: slow per-process leak → no unbounded memory growth on long unattended runs.
    worker_recycle_games: int = _env_int("WORKER_RECYCLE_GAMES", 200)

    # --- primary (Blackwell) GPU leaf-eval server + RL concurrency -----------
    #: Concurrent persistent GPU leaf-eval server replicas on the Blackwell.
    #: Several CUDA contexts run in parallel so one stream's GPU work hides
    #: another's CPU/IPC idle → higher game-phase util. ~1-2 GB each (bf16) → 6
    #: sits well under the 48 GB card's ~40 GB VRAM target.
    leaf_server_replicas: int = _env_int("LEAF_SERVER_REPLICAS", 6)
    #: Max leaves coalesced into ONE server forward. Bounded for VRAM safety;
    #: the server's OomGuard shrinks further on an actual CUDA OOM. Bigger = more
    #: work per IPC roundtrip (fewer, bigger forwards → better GPU efficiency).
    leaf_server_max_batch: int = _env_int("LEAF_SERVER_MAX_BATCH", 1024)
    #: Coalesce wait window (ms): let near-simultaneous worker requests queue so
    #: each forward is large rather than many tiny ones.
    leaf_server_coalesce_ms: float = _env_float("LEAF_SERVER_COALESCE_MS", 4.0)
    #: Concurrent in-flight self-play games for the RL loop WHEN the GPU leaf
    #: server is active. Deliberately OVERSUBSCRIBES the 32 threads so IPC/GPU-
    #: blocked workers overlap compute workers → CPU pegged (the intended bind).
    #: Bounded so per-worker RSS × count stays within the RAM budget with margin.
    rl_games_in_flight: int = _env_int("RL_GAMES_IN_FLIGHT", 40)
    #: Approx resident memory per sim/game worker (cg runtime + torch import +
    #: featurization buffers). Used to cap concurrency so RSS stays in budget.
    per_worker_rss_gb: float = _env_float("PER_WORKER_RSS_GB", 0.8)

    # --- RAM budget (PHYSICAL 128 GB + 256 GB swap as a SOFT overflow) --------
    # CONSOLIDATED stance: the objective is MAX GPU + 32-thread CPU utilisation
    # without crashing; memory knobs are subordinate to that. Swap (256 GB) is a
    # fine overflow cushion — don't fight it — so OOM is not a crash risk. The
    # ONE rule we protect for throughput: the HOT working set (tensors read every
    # training/dataloader step: current batches, prefetch queue, shuffle buffer)
    # stays in physical RAM so swap thrash never starves the GPU. COLD, rarely-
    # touched cache spilling to swap is acceptable. If trading a little cache for
    # more dataloader workers/prefetch keeps the GPU fed, throughput wins.
    #
    #: Target resident budget for the HOT working set (kept in physical RAM).
    #: ~100 GB of 128 GB; the ~28 GB margin covers OS + per-worker process memory
    #: + activation buffers. Cold cache beyond this may spill to swap (fine).
    ram_cache_gb: float = _env_float("RAM_CACHE_GB", 100.0)
    #: Cold data (rarely-touched cache entries) may live in swap-eligible memory.
    #: Kept True per the relaxed stance; set False to force everything resident.
    allow_swap_overflow: bool = _env_bool("ALLOW_SWAP_OVERFLOW", True)
    #: SOFT floor of free physical RAM. When available RAM dips below this, the
    #: (gentle) memory-pressure guard stops *growing* the hot set and lets new /
    #: cold entries be swap-eligible instead of hard-stopping. Not a crash guard
    #: anymore — just keeps the hot set from being paged out. ~8 GB soft cushion.
    free_ram_floor_gb: float = _env_float("FREE_RAM_FLOOR_GB", 8.0)
    #: Streaming shuffle-buffer size (sequences held in RAM at once). Big but
    #: bounded — the corpus is streamed, so this hot buffer is what stays
    #: RAM-resident; it is shrunk gently only under real memory pressure.
    shuffle_buffer_seqs: int = _env_int("SHUFFLE_BUFFER_SEQS", 8192)

    #: RAM-resident cache root. Points at data/cache/ by default; set
    #: POKEBOT_CACHE_DIR to a tmpfs mount (e.g. /dev/shm/pokebot) for true RAM.
    cache_dir: Path = field(default_factory=lambda: _env_path("CACHE_DIR", paths.CACHE_DIR))
    outputs_dir: Path = field(default_factory=lambda: _env_path("OUTPUTS_DIR", paths.OUTPUTS_DIR))
    checkpoints_dir: Path = field(
        default_factory=lambda: _env_path("CHECKPOINTS_DIR", paths.CHECKPOINTS_DIR)
    )

    #: Name substrings used by device.py to pin GPUs.
    train_gpu_name: str = os.environ.get("POKEBOT_TRAIN_GPU_NAME", "Blackwell")
    leaf_gpu_name: str = os.environ.get("POKEBOT_LEAF_GPU_NAME", "3080")

    # --- global perf toggles (applied by apply_runtime_perf) -----------------
    allow_tf32: bool = _env_bool("ALLOW_TF32", True)
    cudnn_benchmark: bool = _env_bool("CUDNN_BENCHMARK", True)
    #: AMP compute dtype for every train/infer autocast (bf16 = fp32 range, no
    #: GradScaler needed; Ampere+Blackwell both support it).
    amp_dtype: str = os.environ.get("POKEBOT_AMP_DTYPE", "bf16")


@dataclass
class ModelConfig:
    """Policy/value transformer knobs.

    The trusted path is history conditioned: each acting seat sees only its own
    deployment-visible sequence of observations. Training consumes the same
    causal sequence that serving consumes incrementally. Counterfactual search
    cannot reuse this realized history without branch-local belief/history
    state, so legacy single-world MCTS is experimental and excluded from
    trusted targets and promotion.
    """

    d_model: int = _env_int("D_MODEL", 256)
    spatial_layers: int = _env_int("SPATIAL_LAYERS", 4)
    temporal_layers: int = _env_int("TEMPORAL_LAYERS", 4)
    option_decoder_layers: int = _env_int("OPTION_DECODER_LAYERS", 2)
    n_heads: int = _env_int("N_HEADS", 8)
    ff_dim: int = _env_int("FF_DIM", 1024)
    #: Realized observation-history ceiling for the temporal tower.
    max_context: int = _env_int("MAX_CONTEXT", 320)
    #: Temporal positional scheme: ``"rope"`` (preferred) or ``"learned"``.
    temporal_pos: str = os.environ.get("POKEBOT_TEMPORAL_POS", "rope")
    #: ``history`` is production-trusted; ``stateless`` is legacy compatibility.
    decision_context: str = os.environ.get("POKEBOT_DECISION_CONTEXT", "history")
    #: Incremental serving cache for the trusted realized-history path.
    kv_cache: bool = _env_bool("KV_CACHE", True)
    #: Previous-own-action embedding residual. Kept conservative so legacy
    #: board-history checkpoints are not destabilized before anchored fine-tune.
    history_action_scale: float = _env_float("HISTORY_ACTION_SCALE", 0.1)
    card_embed_dim: int = _env_int("CARD_EMBED_DIM", 64)
    attack_embed_dim: int = _env_int("ATTACK_EMBED_DIM", 32)
    dropout: float = _env_float("DROPOUT", 0.1)

    @property
    def card_vocab(self) -> int:
        return CARD_VOCAB

    @property
    def attack_vocab(self) -> int:
        return ATTACK_VOCAB


@dataclass
class SearchConfig:
    """Policy-first trusted runtime and experimental MCTS knobs."""

    #: Trusted default is policy-first. ``oracle_mcts`` is diagnostic only and
    #: must never generate promotion/training evidence or ship in a submission.
    mode: str = os.environ.get("POKEBOT_SEARCH_MODE", "policy")
    allow_oracle_deck: bool = _env_bool("ALLOW_ORACLE_DECK", False)

    #: Per-game think budget (competition ladder evidence ~600s / 10 min).
    game_time_budget_s: float = _env_float("GAME_TIME_BUDGET_S", 600.0)
    #: Soft per-move ceiling; verified empirically, not the outdated 1s claim.
    move_time_budget_s: float = _env_float("MOVE_TIME_BUDGET_S", 8.0)
    #: Starting sim count per move (do not hard-cap at the sample's 10).
    sims_per_move: int = _env_int("SIMS_PER_MOVE", 64)
    #: Floor sims even when the move clock is nearly empty.
    min_sims: int = _env_int("MIN_SIMS", 1)
    puct_c: float = _env_float("PUCT_C", 1.25)
    #: Max leaves packed into one GPU forward. Fallback default; the GPU-sized
    #: value from :func:`leaf_batch_for_device` (256 on the 3080 Ti, 512 on the
    #: Blackwell) is preferred when a device is known. Overridable via
    #: ``POKEBOT_LEAF_BATCH_SIZE``.
    leaf_batch_size: int = _env_int("LEAF_BATCH_SIZE", 256)
    #: Micro-batch flush timeout when gathering leaves across trees (ms).
    inference_batch_timeout_ms: float = _env_float("INFERENCE_BATCH_TIMEOUT_MS", 2.0)
    #: Virtual-loss preselection inside one tree is disabled: selecting a blind
    #: wave before seeing earlier leaf values is not adaptive MCTS. GPU batching
    #: still happens across concurrent games in the persistent leaf server.
    leaf_batch_mcts: bool = _env_bool("LEAF_BATCH_MCTS", False)
    #: A remote leaf request that receives no response fails closed after this
    #: bound instead of hanging a worker forever.
    remote_request_timeout_s: float = _env_float("REMOTE_REQUEST_TIMEOUT_S", 30.0)
    #: Search is invalid when materially fewer than the planned simulations
    #: complete. The absolute requirement remains at least one simulation.
    min_sim_completion_ratio: float = _env_float("MIN_SIM_COMPLETION_RATIO", 0.9)
    #: Extra sims multiplier for complex positions (many legal options).
    complex_option_threshold: int = _env_int("COMPLEX_OPTION_THRESHOLD", 8)
    complex_sims_mult: float = _env_float("COMPLEX_SIMS_MULT", 1.5)


@dataclass
class CheckpointConfig:
    """Periodic / resumable checkpointing (:mod:`poke_bot.checkpoint`)."""

    every_steps: int = _env_int("CHECKPOINT_EVERY_STEPS", 500)
    #: Frequent wall-clock checkpoints so an unattended run loses little on a
    #: crash/restart (resumable via --resume auto).
    every_minutes: float = _env_float("CHECKPOINT_EVERY_MINUTES", 5.0)
    keep_last_k: int = _env_int("KEEP_LAST_K", 3)
    #: Default resume mode for train scripts: ``auto`` / ``0`` / path.
    resume: str = os.environ.get("POKEBOT_RESUME", "auto")


@dataclass
class PureRLConfig:
    """Single-trainee pure-RL knobs (core first, full-box hardware)."""

    enabled: bool = _env_bool("PURE_RL", False)
    awr_beta: float = _env_float("PURE_RL_AWR_BETA", 0.5)
    awr_weight_max: float = _env_float("PURE_RL_AWR_WEIGHT_MAX", 20.0)
    bootstrap_mix: float = _env_float("PURE_RL_BOOTSTRAP_MIX", 0.0)
    gate_wr: float = _env_float("PURE_RL_GATE_WR", 0.70)
    min_heldout_games: int = _env_int("PURE_RL_MIN_HELDOUT_GAMES", 200)
    collect_temperature: float = _env_float("PURE_RL_COLLECT_TEMPERATURE", 1.0)
    allow_single_gpu: bool = _env_bool("PURE_RL_ALLOW_SINGLE_GPU", False)


HARDWARE = HardwareConfig()
MODEL = ModelConfig()
SEARCH = SearchConfig()
PURE_RL = PureRLConfig()
CHECKPOINT = CheckpointConfig()


# ===========================================================================
# Per-GPU batch profile — safe-but-aggressive sizing (dual-GPU, unattended)
# ===========================================================================

@dataclass
class BatchProfile:
    """Throughput sizing for one GPU role. Batch = decision-timestep budget.

    Sequences are variable-length whole games, so "batch size" is expressed as
    ``games_per_batch`` (sequences accumulated before an optimizer step) and
    ``max_decisions_per_batch`` (a cap on total decision timesteps per micro-
    batch). ``leaf_batch_size`` is the MCTS/batched-inference GPU forward width.

    ``vram_target_gb`` / ``vram_total_gb`` document the ~85%-of-card target the
    numbers aim for; the real crash-safety net is the OOM catch-and-halve guard
    (:class:`OomGuard`) + ``expandable_segments`` — util is the priority, the
    guard keeps the unattended box alive if a spike exceeds the target.
    """

    key: str
    games_per_batch: int
    max_decisions_per_batch: int
    shuffle_buffer: int
    leaf_batch_size: int
    grad_accum_steps: int = 1
    amp_dtype: str = "bf16"
    vram_target_gb: float = 0.0
    vram_total_gb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


#: RTX PRO 5000 Blackwell (48 GB) — primary archetype train / RL / inference.
#: Targets ~40 GB (~85%). bf16, no grad-accum (big single-step batch).
_BW = BatchProfile(
    key="blackwell",
    games_per_batch=_env_int("BW_GAMES_PER_BATCH", 48),
    max_decisions_per_batch=_env_int("BW_MAX_DECISIONS", 4096),
    shuffle_buffer=_env_int("BW_SHUFFLE_BUFFER", 8192),
    leaf_batch_size=_env_int("BW_LEAF_BATCH", 512),
    grad_accum_steps=_env_int("BW_GRAD_ACCUM", 1),
    amp_dtype=os.environ.get("POKEBOT_BW_AMP", "bf16"),
    vram_target_gb=_env_float("BW_VRAM_TARGET_GB", 40.0),
    vram_total_gb=48.0,
)

#: RTX 3080 Ti (12 GB) — core-kernel / specialists / batched leaf eval.
#: Targets ~10 GB (~85%). bf16. (Probe: the lean d_model=192 kernel peaks ~2 GB,
#: so this is CPU-featurization-bound in practice; sized bold anyway.)
_TI = BatchProfile(
    key="3080ti",
    games_per_batch=_env_int("TI_GAMES_PER_BATCH", 24),
    max_decisions_per_batch=_env_int("TI_MAX_DECISIONS", 2048),
    shuffle_buffer=_env_int("TI_SHUFFLE_BUFFER", 4096),
    leaf_batch_size=_env_int("TI_LEAF_BATCH", 256),
    grad_accum_steps=_env_int("TI_GRAD_ACCUM", 1),
    amp_dtype=os.environ.get("POKEBOT_TI_AMP", "bf16"),
    vram_target_gb=_env_float("TI_VRAM_TARGET_GB", 10.0),
    vram_total_gb=12.0,
)

#: CPU fallback (no CUDA) — small so a torch-less/CI box never explodes.
_CPU = BatchProfile(
    key="cpu",
    games_per_batch=_env_int("CPU_GAMES_PER_BATCH", 4),
    max_decisions_per_batch=_env_int("CPU_MAX_DECISIONS", 256),
    shuffle_buffer=_env_int("CPU_SHUFFLE_BUFFER", 256),
    leaf_batch_size=_env_int("CPU_LEAF_BATCH", 16),
    grad_accum_steps=1,
    amp_dtype="fp32",
    vram_target_gb=0.0,
    vram_total_gb=0.0,
)

_BATCH_PROFILES: dict[str, BatchProfile] = {"blackwell": _BW, "3080ti": _TI, "cpu": _CPU}


def gpu_kind(device: Any = None) -> str:
    """Classify a device → ``"blackwell"`` / ``"3080ti"`` / ``"cpu"``.

    Accepts a ``torch.device``, an int index, a ``"cuda:N"`` / ``"cpu"`` string,
    or ``None`` (auto: prefer Blackwell, else any CUDA, else cpu). torch is
    imported lazily so this stays usable in a torch-less environment.
    """
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"

    idx: Optional[int] = None
    if device is None:
        # Auto: match the training preference (Blackwell) if present.
        for i in range(torch.cuda.device_count()):
            if HARDWARE.train_gpu_name.lower() in torch.cuda.get_device_name(i).lower():
                idx = i
                break
        if idx is None:
            idx = 0
    else:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cpu":
            return "cpu"
        idx = dev.index if dev.index is not None else torch.cuda.current_device()

    try:
        name = torch.cuda.get_device_name(idx).lower()
    except Exception:
        return "cpu"
    if HARDWARE.train_gpu_name.lower() in name or "blackwell" in name:
        return "blackwell"
    if HARDWARE.leaf_gpu_name.lower() in name or "3080" in name:
        return "3080ti"
    # Unknown CUDA card: treat by VRAM (>=24 GB → big/blackwell-like).
    try:
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return "blackwell" if total_gb >= 24 else "3080ti"
    except Exception:
        return "3080ti"


def batch_profile(device_or_kind: Any = None) -> BatchProfile:
    """Return the :class:`BatchProfile` for a device or an explicit kind string."""
    if isinstance(device_or_kind, str) and device_or_kind in _BATCH_PROFILES:
        return _BATCH_PROFILES[device_or_kind]
    return _BATCH_PROFILES[gpu_kind(device_or_kind)]


def leaf_batch_for_device(device: Any = None) -> int:
    """GPU-sized MCTS/batched-inference leaf batch width for ``device``."""
    return batch_profile(device).leaf_batch_size


def dataloader_kwargs(**overrides: Any) -> dict[str, Any]:
    """``torch.utils.data.DataLoader`` kwargs from the profile.

    Keeps the GPU fed: ``num_workers`` loader processes, pinned memory, workers
    persisted across epochs, and a bounded ``prefetch_factor`` (so the prefetch
    queue — part of the HOT set — stays RAM-resident). ``prefetch_factor`` /
    ``persistent_workers`` are only emitted when ``num_workers > 0`` (torch
    rejects them otherwise). Pass overrides (e.g. ``num_workers=0``) freely.
    """
    n = int(overrides.pop("num_workers", HARDWARE.dataloader_workers))
    kw: dict[str, Any] = {
        "num_workers": n,
        "pin_memory": bool(overrides.pop("pin_memory", HARDWARE.pin_memory)),
    }
    if n > 0:
        kw["persistent_workers"] = bool(
            overrides.pop("persistent_workers", HARDWARE.persistent_workers)
        )
        kw["prefetch_factor"] = int(
            overrides.pop("prefetch_factor", HARDWARE.prefetch_factor)
        )
    else:
        overrides.pop("persistent_workers", None)
        overrides.pop("prefetch_factor", None)
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# Global perf toggles (TF32 / cuDNN benchmark / thread counts)
# ---------------------------------------------------------------------------

_PERF_APPLIED = False


def apply_runtime_perf(torch_module: Any = None, *, force: bool = False) -> dict[str, Any]:
    """Apply process-wide perf toggles once. Safe to call from every entry point.

    Sets TF32 matmul + cuDNN allow_tf32, ``cudnn.benchmark`` (autotune kernels
    for our fixed shapes), a high fp32 matmul precision, and pins intra-op
    threads to :attr:`HARDWARE.torch_threads`. Idempotent (guarded) unless
    ``force``. Returns a dict describing what was applied (for dry-run logging).
    """
    global _PERF_APPLIED
    if _PERF_APPLIED and not force:
        return {"applied": False, "reason": "already applied"}
    try:
        torch = torch_module or __import__("torch")
    except Exception as exc:  # torch-less environment — nothing to do.
        return {"applied": False, "reason": f"torch unavailable: {exc}"}

    applied: dict[str, Any] = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = HARDWARE.allow_tf32
        torch.backends.cudnn.allow_tf32 = HARDWARE.allow_tf32
        applied["allow_tf32"] = HARDWARE.allow_tf32
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = HARDWARE.cudnn_benchmark
        applied["cudnn_benchmark"] = HARDWARE.cudnn_benchmark
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
        applied["float32_matmul_precision"] = "high"
    except Exception:
        pass
    try:
        torch.set_num_threads(int(HARDWARE.torch_threads))
        applied["torch_num_threads"] = int(HARDWARE.torch_threads)
    except Exception:
        pass
    _PERF_APPLIED = True
    return {"applied": True, **applied}


# ---------------------------------------------------------------------------
# Memory-pressure (RAM) + CUDA OOM guards — unattended crash-safety
# ---------------------------------------------------------------------------

def memory_pressure() -> dict[str, Any]:
    """Snapshot host-RAM pressure (psutil if present, else /proc/meminfo).

    Returns ``{available_gb, total_gb, used_percent, under_floor}`` where
    ``under_floor`` is True when free RAM has dropped below
    :attr:`HARDWARE.free_ram_floor_gb`. Gentle: this is advisory (swap is an
    accepted overflow), used to stop *growing* the hot set — not a hard stop.
    """
    avail_gb = total_gb = 0.0
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1e9
        total_gb = vm.total / 1e9
    except Exception:
        try:
            meta: dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    k, _, rest = line.partition(":")
                    meta[k.strip()] = int(rest.strip().split()[0])  # kB
            total_gb = meta.get("MemTotal", 0) / 1e6
            avail_gb = meta.get("MemAvailable", meta.get("MemFree", 0)) / 1e6
        except Exception:
            return {"available_gb": 0.0, "total_gb": 0.0, "used_percent": 0.0,
                    "under_floor": False, "ok": False}
    used_pct = (1.0 - (avail_gb / total_gb)) * 100.0 if total_gb else 0.0
    return {
        "available_gb": round(avail_gb, 2),
        "total_gb": round(total_gb, 2),
        "used_percent": round(used_pct, 1),
        "under_floor": bool(avail_gb and avail_gb < HARDWARE.free_ram_floor_gb),
        "ok": True,
    }


def ram_headroom_ok() -> bool:
    """True if it is safe to KEEP GROWING the hot RAM set (free > soft floor)."""
    mp = memory_pressure()
    return not mp.get("under_floor", False)


def is_cuda_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory RuntimeError (name/message match)."""
    if exc is None:
        return False
    if type(exc).__name__ == "OutOfMemoryError":  # torch.cuda.OutOfMemoryError
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class OomGuard:
    """CUDA-OOM catch → empty_cache → halve batch → continue (unattended-safe).

    Bias is toward utilisation: start at full ``scale`` (1.0) and only shrink on
    an actual OOM, down to ``min_scale``. Use :meth:`scaled` to size a batch and
    :meth:`handle_oom` in an ``except`` block::

        guard = OomGuard()
        for batch in batches:
            try:
                run_step(batch[: guard.scaled(len(batch))])
            except Exception as exc:
                if not guard.handle_oom(exc):
                    raise                # not OOM, or already at the floor
                continue                 # shrank + freed cache → next batch
    """

    def __init__(self, *, min_scale: float = 0.125) -> None:
        self.scale = 1.0
        self.min_scale = float(min_scale)
        self.oom_events = 0

    def scaled(self, n: int) -> int:
        """Current batch size for ``n`` items under the active scale (>=1)."""
        return max(1, int(round(n * self.scale)))

    def handle_oom(self, exc: BaseException, torch_module: Any = None) -> bool:
        """If ``exc`` is CUDA OOM: free cache, halve scale, return True (retry).

        Returns False when ``exc`` is not an OOM (caller should re-raise) or the
        scale is already at ``min_scale`` (give up shrinking; caller re-raises).
        """
        if not is_cuda_oom(exc):
            return False
        self.oom_events += 1
        try:
            torch = torch_module or __import__("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if self.scale <= self.min_scale:
            return False
        self.scale = max(self.min_scale, self.scale * 0.5)
        return True


def hardware_profile(device: Any = None) -> dict[str, Any]:
    """One resolved snapshot of the active profile (for logging / dry-runs)."""
    bp = batch_profile(device)
    return {
        "cpu": {
            "cpu_threads": CPU_THREADS,
            "sim_workers": HARDWARE.sim_workers,
            "feature_workers": HARDWARE.feature_workers,
            "torch_threads": HARDWARE.torch_threads,
            "parallel_games": HARDWARE.parallel_games,
        },
        "dataloader": dataloader_kwargs(),
        "ram": {
            "ram_cache_gb": HARDWARE.ram_cache_gb,
            "free_ram_floor_gb": HARDWARE.free_ram_floor_gb,
            "allow_swap_overflow": HARDWARE.allow_swap_overflow,
            "shuffle_buffer_seqs": HARDWARE.shuffle_buffer_seqs,
            "pressure": memory_pressure(),
        },
        "gpu": {
            "kind": gpu_kind(device),
            "batch_profile": bp.as_dict(),
        },
        "perf": {
            "allow_tf32": HARDWARE.allow_tf32,
            "cudnn_benchmark": HARDWARE.cudnn_benchmark,
            "amp_dtype": HARDWARE.amp_dtype,
            "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        },
    }