"""Project configuration: hardware defaults, cache dirs, and model knobs.

Values are read once at import from module-level defaults, each overridable via
an environment variable (``POKEBOT_<NAME>``) so scripts can tune without code
edits. Later phases (dataset/train/self_play/mcts) import these constants.

Hardware target (saturate the box):
  CPU   Ryzen 9 7950X (32 threads)      -> CABT workers + MCTS sim threads
  RAM   128 GB (256 GB swap safety)     -> RAM-resident replay/tensor cache
  GPU0  RTX 3080 Ti                     -> batched MCTS leaf eval
  GPU1  RTX PRO 5000 Blackwell          -> primary training + batched inference
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths


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


@dataclass
class HardwareConfig:
    """CPU / worker / cache defaults."""

    sim_workers: int = _env_int("SIM_WORKERS", 28)
    feature_workers: int = _env_int("FEATURE_WORKERS", 30)
    dataloader_workers: int = _env_int("DATALOADER_WORKERS", 8)
    pin_memory: bool = _env_bool("PIN_MEMORY", True)

    #: In-process concurrent MCTS trees / games that share one leaf batcher.
    #: libcg battle is still one-at-a-time per process; this sizes multi-tree
    #: search + reanalyse collect (see ``poke_bot.self_play``).
    parallel_games: int = _env_int("PARALLEL_GAMES", 8)

    #: Recycle each sim worker process after this many games (libcg leaks slowly).
    worker_recycle_games: int = _env_int("WORKER_RECYCLE_GAMES", 200)

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


@dataclass
class ModelConfig:
    """Temporal transformer knobs (v1 Hammer submit sizing).

    Consumed by :mod:`poke_bot.model`; features.py and config stay the single
    source of truth for dimensions.

    Whole-game context: ``max_context`` caps decision timesteps (not a short
    sliding window). Temporal positions use RoPE; inference keeps a KV cache.
    """

    d_model: int = _env_int("D_MODEL", 256)
    spatial_layers: int = _env_int("SPATIAL_LAYERS", 4)
    temporal_layers: int = _env_int("TEMPORAL_LAYERS", 4)
    option_decoder_layers: int = _env_int("OPTION_DECODER_LAYERS", 2)
    n_heads: int = _env_int("N_HEADS", 8)
    ff_dim: int = _env_int("FF_DIM", 1024)
    #: Max decision timesteps kept in the causal temporal tower (whole-game).
    #: Empirically: p99 per-seat decisions ≈ 309 on 2026-07-12 ladder episodes
    #: (4985 games / 9970 seats); 320 covers ≥99.15%. See outputs/notes/max_context.md.
    max_context: int = _env_int("MAX_CONTEXT", 320)
    #: Temporal positional scheme: ``"rope"`` (preferred) or ``"learned"``.
    temporal_pos: str = os.environ.get("POKEBOT_TEMPORAL_POS", "rope")
    #: Incremental encode via KV cache (append one [CLS] per realized decision).
    kv_cache: bool = _env_bool("KV_CACHE", True)
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
    """MCTS / search defaults (:mod:`poke_bot.mcts`)."""

    #: Per-game think budget (competition ladder evidence ~600s / 10 min).
    game_time_budget_s: float = _env_float("GAME_TIME_BUDGET_S", 600.0)
    #: Soft per-move ceiling; verified empirically, not the outdated 1s claim.
    move_time_budget_s: float = _env_float("MOVE_TIME_BUDGET_S", 8.0)
    #: Starting sim count per move (do not hard-cap at the sample's 10).
    sims_per_move: int = _env_int("SIMS_PER_MOVE", 64)
    #: Floor sims even when the move clock is nearly empty.
    min_sims: int = _env_int("MIN_SIMS", 1)
    puct_c: float = _env_float("PUCT_C", 1.25)
    #: Max leaves packed into one GPU forward (aligns with ``POKEBOT_LEAF_BATCH_SIZE``).
    leaf_batch_size: int = _env_int("LEAF_BATCH_SIZE", 64)
    #: Micro-batch flush timeout when gathering leaves across trees (ms).
    inference_batch_timeout_ms: float = _env_float("INFERENCE_BATCH_TIMEOUT_MS", 2.0)
    #: Use virtual-loss multi-leaf batching inside a single MCTS search.
    leaf_batch_mcts: bool = _env_bool("LEAF_BATCH_MCTS", True)
    #: Extra sims multiplier for complex positions (many legal options).
    complex_option_threshold: int = _env_int("COMPLEX_OPTION_THRESHOLD", 8)
    complex_sims_mult: float = _env_float("COMPLEX_SIMS_MULT", 1.5)


@dataclass
class CheckpointConfig:
    """Periodic / resumable checkpointing (:mod:`poke_bot.checkpoint`)."""

    every_steps: int = _env_int("CHECKPOINT_EVERY_STEPS", 500)
    every_minutes: float = _env_float("CHECKPOINT_EVERY_MINUTES", 10.0)
    keep_last_k: int = _env_int("KEEP_LAST_K", 3)
    #: Default resume mode for train scripts: ``auto`` / ``0`` / path.
    resume: str = os.environ.get("POKEBOT_RESUME", "auto")


HARDWARE = HardwareConfig()
MODEL = ModelConfig()
SEARCH = SearchConfig()
CHECKPOINT = CheckpointConfig()