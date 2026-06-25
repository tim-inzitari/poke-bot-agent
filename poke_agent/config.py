from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from poke_agent.features import FEATURE_SCHEMA_VERSION, STRUCTURED_FEATURE_DIM, total_feature_dim
from poke_agent.outputs import describe_layout, ensure_output_layout, report_path, resolve_checkpoint_path

# =============================================================================
# Edit these variables to configure training, data paths, and model settings.
# Used by the notebook, scripts/train_agent.py, and python -m poke_agent.main.
# Environment variables override individual values when set (see _ENV_MAP below).
# =============================================================================

# --- Data paths ---
# AGENT_DECK_PATH is the deck we submit to Kaggle (hard-played at runtime).
PRIMARY_ROLLOUT_DATA = "data/training_rollouts_merged.jsonl"
SCRAPED_ROLLOUT_DATA = "data/scraped_rollouts.jsonl"
MULTIDECK_ROLLOUT_DATA = "data/multideck_rollouts.jsonl"
MERGED_ROLLOUT_DATA = "data/training_rollouts_merged.jsonl"
EPISODES_INDEX_PATH = "kaggle/input/pokemon-tcg-ai-battle-episodes-index/manifest.csv"
# Top N%% of episodes within each loaded daily bundle (by replay score). 1.0 = top 1%%.
TOP_EPISODE_PERCENT = 1.0
TRAINING_DECK_DIRS = ["decks/archetype-samples"]
TRAINING_ROLLOUT_SOURCES = [
    "data/scraped_rollouts.jsonl",
    "data/multideck_rollouts.jsonl",
]
REQUIRE_COMPLETE_GAMES = True
REQUIRE_TRAINING_MATCHUP_DIVERSITY = True
MIN_TRAINING_MATCHUPS = 2
MIN_TRAINING_DECK_SLUGS = 2
# Bootstrap data quality: require competition replay games from the official
# pokemon-tcg-ai-battle-episodes-index dataset (not synthetic multideck only).
REQUIRE_TOP_OF_LADDER_DATA = True
MIN_TOP_OF_LADDER_FRACTION = 0.0
FALLBACK_ROLLOUT_DATA = [
    "data/mac-rollouts-10k.jsonl",
    "data/notebook_rollouts.jsonl",
    "data/kaggle-output/data/cabt_rollouts.jsonl",
    "data/deckpool-smoke.jsonl",
    "data/container-mp-smoke.jsonl",
    "data/container-smoke.jsonl",
]
AGENT_DECK_PATH = (
    "decks/competitive/high_performing/"
    "2026-05_regional-melbourne-2026_10th_mega-lucario.csv"
)
CABT_GENERATED_PATH = "data/multideck_rollouts.jsonl"
COMPETITION_RESULTS_PATH = "data/competition-results.jsonl"
MODEL_ID = "temporal_current"
MODEL_OUTPUT_PATH = "outputs/checkpoints/temporal_current.pt"
REQUIRE_CABT_EVAL_DATA = True

# --- Dataset size (games) ---
DATASET_GAMES = 2000  # None = train on all games in file, skip inline generation
                      # 5000 or 100000 = that many CABT games to generate and/or cap training

# --- Features ---
TRANSITION_CLASSES = 64
STATE_HASH_DIM = 32
WINDOW_SIZE = 256
CARD_VOCAB_SIZE = 2000
CARD_EMBED_DIM = 32
TENSOR_BUILD_WORKERS = None  # None = cpu_count - 2 (e.g. 30 on a 32-thread CPU)

# --- Model ---
MODEL_D_MODEL = 256
MODEL_HEADS = 8
MODEL_LAYERS = 6
MODEL_FF = 1024
MODEL_DROPOUT = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

# --- Loss weights ---
LOSS_VALUE_WEIGHT = 1.0
LOSS_POLICY_WEIGHT = 0.85
LOSS_DYNAMICS_WEIGHT = 0.15
LOSS_ENTROPY_WEIGHT = 0.01
LOSS_UNCERTAINTY_WEIGHT = 0.02

# --- Rewards (training labels) ---
# Win = YOUR prize cards reach 0 (take prizes when you KO opponent Pokémon).
VALUE_WIN = 1.0
VALUE_NOT_WIN = -1.0  # loss or draw
VALUE_TIMEOUT = -2.0  # harsh penalty for stalling to the time limit
VALUE_RETURN_GAMMA = 0.997
VALUE_SHAPING_ALPHA = 0.15
VALUE_LAMBDA = 0.9
VALUE_MC_BLEND = 0.6
POLICY_WEIGHTING = "awr"  # awr | none
POLICY_AWR_BETA = 0.5
POLICY_AWR_WEIGHT_MAX = 20.0
POLICY_SOFT_TOPK = 8
OBJECTIVE_USE_SOFT_SEARCH_POLICY = True
OBJECTIVE_SEARCH_POLICY_KL_WEIGHT = 1.0
TRAIN_ENCODER_LR = 1e-4
TRAIN_HEAD_LR = 3e-4

# --- Beam search (Kaggle inference) ---
BEAM_WIDTH = 12
BEAM_TIME_BUDGET_MS = 5000
BEAM_MIN_REMAINING_SEC = 120
# Self-play: sim beam used during baseline/self-play collection and eval.
SELF_PLAY_BEAM_WIDTH = 8
SELF_PLAY_BEAM_TIME_BUDGET_MS = 1500
SELF_PLAY_BEAM_MAX_SEARCH_STEPS = 128
SELF_PLAY_BEAM_ROLLOUT_POLICY_WIDTH = 12
SEARCH_DETERMINIZATIONS = 2
COLLECTION_INFERENCE_DEVICE = "auto"  # auto | cpu | cuda

# --- Self-play (AlphaGo-style loop) ---
# One "iteration" = play SELF_PLAY_GAMES cabt games, then retrain on ALL self-play JSONL so far.
# SELF_PLAY_ITERATIONS = max collect→train cycles (stops early on plateau or target win rate).
SELF_PLAY_GAMES = 20  # CABT games per iteration (your deck vs field + past checkpoints)
SELF_PLAY_ITERATIONS = 100
SELF_PLAY_TRAIN_EPOCHS = 500  # transformer epochs after each self-play iteration
SELF_PLAY_EVAL_GAMES = 20
SELF_PLAY_OPPONENT_POOL_SIZE = 16
SELF_PLAY_USE_BEAM = True
SELF_PLAY_OUTPUT_PATH = "outputs/rollouts/self_play_rollouts.jsonl"
SELF_PLAY_CHECKPOINT_DIR = "outputs/checkpoints/self_play"
SELF_PLAY_TRAIN_AFTER_COLLECT = True  # retrain every iteration on accumulated self-play data
SELF_PLAY_FIELD_DECK_DIR = "decks/competitive/high_performing"
SELF_PLAY_MATCHUP_MODE = "sample"  # sample | round-robin
SELF_PLAY_TARGET_RANK = 1000  # eval/bar: win vs opponent decks with placement <= this
SELF_PLAY_TARGET_WIN_RATE = 0.55
SELF_PLAY_PLATEAU_PATIENCE = 5
SELF_PLAY_WORKERS = 10  # parallel CABT games per iteration (baseline + self-play collect)
SELF_PLAY_OPPONENT_LATEST_PROB = 0.6  # PFSP-lite: fraction of games vs latest checkpoint
# Official Kaggle sample agents (main.py + deck.csv each) — see baselines/README.md
SELF_PLAY_BASELINE_DIR = "baselines/official"
SELF_PLAY_BASELINE_WIN_RATE = 0.60  # aggregate vs all baselines → start transformer self-play
SELF_PLAY_BASELINE_ARCHETYPE_DECKS_ONLY = True  # our-side decks = high-performing baseline archetype lists
SELF_PLAY_BASELINE_TOP_DECKS_PER_ARCHETYPE = 3  # best N lists per baseline agent (by event placement)
SELF_PLAY_AGENT_DECK_DIR = "decks/archetype-samples"  # used when BASELINE_ARCHETYPE_DECKS_ONLY=0
SELF_PLAY_PER_DECK_CHECKPOINT_DIR = None  # None = skip per-deck checkpoint copies (saves disk)
SELF_PLAY_TRAIN_WINDOW_GAMES = None  # None = match SELF_PLAY_GAMES (--games per iteration)
SELF_PLAY_TRIM_ROLLOUT_FILE = True  # keep JSONL on disk trimmed to the train window
SELF_PLAY_WARMUP_ITERATIONS = 10  # first N transformer self-play cycles use boosted LR (0 = off)
SELF_PLAY_WARMUP_LR_MULTIPLIER = 25.0  # LR multiplier during warmup (3e-4 → 7.5e-3 at default)

# --- Training loop ---
TRAIN_EPOCHS = 1000
# Stop when value_loss (win/loss prediction) plateaus — not on tiny total-loss noise.
EARLY_STOP_METRIC = "value_loss"  # value_loss | total_loss
EARLY_STOP_PATIENCE = 20  # epochs without meaningful improvement → stop
EARLY_STOP_MIN_DELTA = 5e-4  # minimum drop on monitor metric to count as improvement
TRAIN_PRINT_EVERY = 100
BATCH_GAMES = 4  # seat-sequences per training batch
TRAIN_USE_AMP = True
TRAIN_GRAD_CHECKPOINT = False
# Where the full dataset tensors live. "cpu" keeps them in system RAM and streams
# each batch to the GPU (lets you train on far more games than fit in VRAM, bounded
# by RAM instead). "cuda"/"device" keeps the whole dataset resident on the GPU
# (fastest, but caps dataset size to VRAM). "auto" = cpu when training on CUDA.
TRAIN_DATA_DEVICE = "auto"
# Crash/OOM resilience: write a resumable checkpoint every N epochs (0 = off) plus
# a best-so-far copy. Files sit beside the final checkpoint as <name>.latest.pt /
# <name>.best.pt. TRAIN_RESUME: "auto" resumes <name>.latest.pt if present,
# "0" never resumes, "1" requires it.
TRAIN_CHECKPOINT_EVERY = 5
TRAIN_RESUME = "auto"
# Persist built training tensors to disk so resume skips JSONL parse + feature build.
# auto = load cache when valid, save after build; 0 = off; 1 = require cache; rebuild = ignore.
TRAIN_TENSOR_CACHE = "auto"
TRAIN_TENSOR_CACHE_DIR = "outputs/cache/training_tensors"

# Flat override keys accepted by build_config(..., overrides=...).
OVERRIDE_KEYS = frozenset({
    "primary_rollout_data",
    "fallback_rollout_data",
    "agent_deck_path",
    "cabt_generated_path",
    "competition_results_path",
    "model_output_path",
    "model_id",
    "require_cabt_eval_data",
    "dataset_games",
    "transition_classes",
    "state_hash_dim",
    "window_size",
    "card_vocab_size",
    "card_embed_dim",
    "value_return_gamma",
    "value_shaping_alpha",
    "value_lambda",
    "value_mc_blend",
    "policy_weighting",
    "policy_awr_beta",
    "policy_awr_weight_max",
    "policy_soft_topk",
    "objective_use_soft_search_policy",
    "objective_search_policy_kl_weight",
    "train_encoder_lr",
    "train_head_lr",
    "train_use_amp",
    "train_grad_checkpoint",
    "train_checkpoint_every",
    "train_resume",
    "train_data_device",
    "train_tensor_cache",
    "train_tensor_cache_dir",
    "d_model",
    "model_heads",
    "model_layers",
    "model_ff",
    "model_dropout",
    "learning_rate",
    "weight_decay",
    "loss_value_weight",
    "loss_policy_weight",
    "loss_dynamics_weight",
    "loss_entropy_weight",
    "loss_uncertainty_weight",
    "train_epochs",
    "early_stop_patience",
    "early_stop_min_delta",
    "early_stop_metric",
    "train_print_every",
    "batch_games",
    "tensor_build_workers",
    "value_win",
    "value_not_win",
    "beam_width",
    "beam_time_budget_ms",
    "beam_min_remaining_sec",
    "self_play_beam_width",
    "self_play_beam_time_budget_ms",
    "self_play_beam_max_search_steps",
    "self_play_beam_rollout_policy_width",
    "search_determinizations",
    "collection_inference_device",
    "self_play_opponent_latest_prob",
    "self_play_games",
    "self_play_iterations",
    "self_play_train_epochs",
    "self_play_eval_games",
    "self_play_opponent_pool_size",
    "self_play_use_beam",
    "self_play_output_path",
    "self_play_checkpoint_dir",
    "self_play_train_after_collect",
    "self_play_field_deck_dir",
    "self_play_matchup_mode",
    "self_play_target_rank",
    "self_play_target_win_rate",
    "self_play_plateau_patience",
    "self_play_workers",
    "self_play_baseline_dir",
    "self_play_baseline_win_rate",
    "self_play_baseline_archetype_decks_only",
    "self_play_baseline_top_decks_per_archetype",
    "self_play_agent_deck_dir",
    "self_play_per_deck_checkpoint_dir",
    "self_play_train_window_games",
    "self_play_trim_rollout_file",
    "self_play_warmup_iterations",
    "self_play_warmup_lr_multiplier",
    "scraped_rollout_data",
    "multideck_rollout_data",
    "merged_rollout_data",
    "training_rollout_sources",
    "episodes_index_path",
    "top_episode_percent",
    "training_deck_dirs",
    "require_complete_games",
    "require_training_matchup_diversity",
    "min_training_matchups",
    "min_training_deck_slugs",
    "require_top_of_ladder_data",
    "min_top_of_ladder_fraction",
    "value_timeout",
})


def default_user_config() -> dict[str, Any]:
    """Return editable defaults as a flat dict for notebooks and scripts."""
    return {
        "primary_rollout_data": PRIMARY_ROLLOUT_DATA,
        "fallback_rollout_data": list(FALLBACK_ROLLOUT_DATA),
        "agent_deck_path": AGENT_DECK_PATH,
        "cabt_generated_path": CABT_GENERATED_PATH,
        "competition_results_path": COMPETITION_RESULTS_PATH,
        "model_output_path": MODEL_OUTPUT_PATH,
        "model_id": MODEL_ID,
        "require_cabt_eval_data": REQUIRE_CABT_EVAL_DATA,
        "dataset_games": DATASET_GAMES,
        "transition_classes": TRANSITION_CLASSES,
        "state_hash_dim": STATE_HASH_DIM,
        "window_size": WINDOW_SIZE,
        "card_vocab_size": CARD_VOCAB_SIZE,
        "card_embed_dim": CARD_EMBED_DIM,
        "value_return_gamma": VALUE_RETURN_GAMMA,
        "value_shaping_alpha": VALUE_SHAPING_ALPHA,
        "value_lambda": VALUE_LAMBDA,
        "value_mc_blend": VALUE_MC_BLEND,
        "policy_weighting": POLICY_WEIGHTING,
        "policy_awr_beta": POLICY_AWR_BETA,
        "policy_awr_weight_max": POLICY_AWR_WEIGHT_MAX,
        "policy_soft_topk": POLICY_SOFT_TOPK,
        "objective_use_soft_search_policy": OBJECTIVE_USE_SOFT_SEARCH_POLICY,
        "objective_search_policy_kl_weight": OBJECTIVE_SEARCH_POLICY_KL_WEIGHT,
        "train_encoder_lr": TRAIN_ENCODER_LR,
        "train_head_lr": TRAIN_HEAD_LR,
        "train_use_amp": TRAIN_USE_AMP,
        "train_grad_checkpoint": TRAIN_GRAD_CHECKPOINT,
        "train_checkpoint_every": TRAIN_CHECKPOINT_EVERY,
        "train_resume": TRAIN_RESUME,
        "train_data_device": TRAIN_DATA_DEVICE,
        "train_tensor_cache": TRAIN_TENSOR_CACHE,
        "train_tensor_cache_dir": TRAIN_TENSOR_CACHE_DIR,
        "d_model": MODEL_D_MODEL,
        "model_heads": MODEL_HEADS,
        "model_layers": MODEL_LAYERS,
        "model_ff": MODEL_FF,
        "model_dropout": MODEL_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss_value_weight": LOSS_VALUE_WEIGHT,
        "loss_policy_weight": LOSS_POLICY_WEIGHT,
        "loss_dynamics_weight": LOSS_DYNAMICS_WEIGHT,
        "loss_entropy_weight": LOSS_ENTROPY_WEIGHT,
        "loss_uncertainty_weight": LOSS_UNCERTAINTY_WEIGHT,
        "train_epochs": TRAIN_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "early_stop_metric": EARLY_STOP_METRIC,
        "train_print_every": TRAIN_PRINT_EVERY,
        "batch_games": BATCH_GAMES,
        "tensor_build_workers": TENSOR_BUILD_WORKERS,
        "value_win": VALUE_WIN,
        "value_not_win": VALUE_NOT_WIN,
        "value_timeout": VALUE_TIMEOUT,
        "beam_width": BEAM_WIDTH,
        "beam_time_budget_ms": BEAM_TIME_BUDGET_MS,
        "beam_min_remaining_sec": BEAM_MIN_REMAINING_SEC,
        "self_play_beam_width": SELF_PLAY_BEAM_WIDTH,
        "self_play_beam_time_budget_ms": SELF_PLAY_BEAM_TIME_BUDGET_MS,
        "self_play_beam_max_search_steps": SELF_PLAY_BEAM_MAX_SEARCH_STEPS,
        "self_play_beam_rollout_policy_width": SELF_PLAY_BEAM_ROLLOUT_POLICY_WIDTH,
        "search_determinizations": SEARCH_DETERMINIZATIONS,
        "collection_inference_device": COLLECTION_INFERENCE_DEVICE,
        "self_play_games": SELF_PLAY_GAMES,
        "self_play_iterations": SELF_PLAY_ITERATIONS,
        "self_play_train_epochs": SELF_PLAY_TRAIN_EPOCHS,
        "self_play_eval_games": SELF_PLAY_EVAL_GAMES,
        "self_play_opponent_pool_size": SELF_PLAY_OPPONENT_POOL_SIZE,
        "self_play_opponent_latest_prob": SELF_PLAY_OPPONENT_LATEST_PROB,
        "self_play_use_beam": SELF_PLAY_USE_BEAM,
        "self_play_output_path": SELF_PLAY_OUTPUT_PATH,
        "self_play_checkpoint_dir": SELF_PLAY_CHECKPOINT_DIR,
        "self_play_train_after_collect": SELF_PLAY_TRAIN_AFTER_COLLECT,
        "self_play_field_deck_dir": SELF_PLAY_FIELD_DECK_DIR,
        "self_play_matchup_mode": SELF_PLAY_MATCHUP_MODE,
        "self_play_target_rank": SELF_PLAY_TARGET_RANK,
        "self_play_target_win_rate": SELF_PLAY_TARGET_WIN_RATE,
        "self_play_plateau_patience": SELF_PLAY_PLATEAU_PATIENCE,
        "self_play_workers": SELF_PLAY_WORKERS,
        "self_play_baseline_dir": SELF_PLAY_BASELINE_DIR,
        "self_play_baseline_win_rate": SELF_PLAY_BASELINE_WIN_RATE,
        "self_play_baseline_archetype_decks_only": SELF_PLAY_BASELINE_ARCHETYPE_DECKS_ONLY,
        "self_play_baseline_top_decks_per_archetype": SELF_PLAY_BASELINE_TOP_DECKS_PER_ARCHETYPE,
        "self_play_agent_deck_dir": SELF_PLAY_AGENT_DECK_DIR,
        "self_play_per_deck_checkpoint_dir": SELF_PLAY_PER_DECK_CHECKPOINT_DIR,
        "self_play_train_window_games": SELF_PLAY_TRAIN_WINDOW_GAMES,
        "self_play_trim_rollout_file": SELF_PLAY_TRIM_ROLLOUT_FILE,
        "self_play_warmup_iterations": SELF_PLAY_WARMUP_ITERATIONS,
        "self_play_warmup_lr_multiplier": SELF_PLAY_WARMUP_LR_MULTIPLIER,
        "scraped_rollout_data": SCRAPED_ROLLOUT_DATA,
        "multideck_rollout_data": MULTIDECK_ROLLOUT_DATA,
        "merged_rollout_data": MERGED_ROLLOUT_DATA,
        "training_rollout_sources": list(TRAINING_ROLLOUT_SOURCES),
        "episodes_index_path": EPISODES_INDEX_PATH,
        "top_episode_percent": TOP_EPISODE_PERCENT,
        "training_deck_dirs": list(TRAINING_DECK_DIRS),
        "require_complete_games": REQUIRE_COMPLETE_GAMES,
        "require_training_matchup_diversity": REQUIRE_TRAINING_MATCHUP_DIVERSITY,
        "min_training_matchups": MIN_TRAINING_MATCHUPS,
        "min_training_deck_slugs": MIN_TRAINING_DECK_SLUGS,
        "require_top_of_ladder_data": REQUIRE_TOP_OF_LADDER_DATA,
        "min_top_of_ladder_fraction": MIN_TOP_OF_LADDER_FRACTION,
    }


_ENV_MAP = {
    "primary_rollout_data": "PRIMARY_ROLLOUT_DATA",
    "agent_deck_path": "AGENT_DECK_PATH",
    "cabt_generated_path": "CABT_GENERATED_PATH",
    "competition_results_path": "COMPETITION_RESULTS_PATH",
    "model_output_path": "MODEL_OUTPUT_PATH",
    "model_id": "MODEL_ID",
    "dataset_games": "DATASET_GAMES",
    "transition_classes": "TRANSITION_CLASSES",
    "state_hash_dim": "STATE_HASH_DIM",
    "window_size": "WINDOW_SIZE",
    "card_vocab_size": "CARD_VOCAB_SIZE",
    "card_embed_dim": "CARD_EMBED_DIM",
    "value_return_gamma": "VALUE_RETURN_GAMMA",
    "value_shaping_alpha": "VALUE_SHAPING_ALPHA",
    "value_lambda": "VALUE_LAMBDA",
    "value_mc_blend": "VALUE_MC_BLEND",
    "policy_weighting": "POLICY_WEIGHTING",
    "policy_awr_beta": "POLICY_AWR_BETA",
    "policy_awr_weight_max": "POLICY_AWR_WEIGHT_MAX",
    "policy_soft_topk": "POLICY_SOFT_TOPK",
    "objective_use_soft_search_policy": "OBJECTIVE_USE_SOFT_SEARCH_POLICY",
    "objective_search_policy_kl_weight": "OBJECTIVE_SEARCH_POLICY_KL_WEIGHT",
    "train_encoder_lr": "TRAIN_ENCODER_LR",
    "train_head_lr": "TRAIN_HEAD_LR",
    "train_use_amp": "TRAIN_USE_AMP",
    "train_grad_checkpoint": "TRAIN_GRAD_CHECKPOINT",
    "train_checkpoint_every": "TRAIN_CHECKPOINT_EVERY",
    "train_resume": "TRAIN_RESUME",
    "train_data_device": "TRAIN_DATA_DEVICE",
    "train_tensor_cache": "TRAIN_TENSOR_CACHE",
    "train_tensor_cache_dir": "TRAIN_TENSOR_CACHE_DIR",
    "top_episode_percent": "TOP_EPISODE_PERCENT",
    "require_top_of_ladder_data": "REQUIRE_TOP_OF_LADDER_DATA",
    "min_top_of_ladder_fraction": "MIN_TOP_OF_LADDER_FRACTION",
    "d_model": "MODEL_D_MODEL",
    "model_heads": "MODEL_HEADS",
    "model_layers": "MODEL_LAYERS",
    "model_ff": "MODEL_FF",
    "model_dropout": "MODEL_DROPOUT",
    "learning_rate": "LEARNING_RATE",
    "weight_decay": "WEIGHT_DECAY",
    "loss_value_weight": "LOSS_VALUE_WEIGHT",
    "loss_policy_weight": "LOSS_POLICY_WEIGHT",
    "loss_dynamics_weight": "LOSS_DYNAMICS_WEIGHT",
    "loss_entropy_weight": "LOSS_ENTROPY_WEIGHT",
    "loss_uncertainty_weight": "LOSS_UNCERTAINTY_WEIGHT",
    "train_epochs": "TRAIN_EPOCHS",
    "early_stop_patience": "EARLY_STOP_PATIENCE",
    "early_stop_min_delta": "EARLY_STOP_MIN_DELTA",
    "early_stop_metric": "EARLY_STOP_METRIC",
    "train_print_every": "TRAIN_PRINT_EVERY",
    "batch_games": "BATCH_GAMES",
    "tensor_build_workers": "TENSOR_BUILD_WORKERS",
    "value_win": "VALUE_WIN",
    "value_not_win": "VALUE_NOT_WIN",
    "value_timeout": "VALUE_TIMEOUT",
    "beam_width": "BEAM_WIDTH",
    "beam_time_budget_ms": "BEAM_TIME_BUDGET_MS",
    "beam_min_remaining_sec": "BEAM_MIN_REMAINING_SEC",
    "self_play_beam_width": "SELF_PLAY_BEAM_WIDTH",
    "self_play_beam_time_budget_ms": "SELF_PLAY_BEAM_TIME_BUDGET_MS",
    "self_play_beam_max_search_steps": "SELF_PLAY_BEAM_MAX_SEARCH_STEPS",
    "self_play_beam_rollout_policy_width": "SELF_PLAY_BEAM_ROLLOUT_POLICY_WIDTH",
    "search_determinizations": "SEARCH_DETERMINIZATIONS",
    "collection_inference_device": "COLLECTION_INFERENCE_DEVICE",
    "self_play_opponent_latest_prob": "SELF_PLAY_OPPONENT_LATEST_PROB",
    "self_play_games": "SELF_PLAY_GAMES",
    "self_play_iterations": "SELF_PLAY_ITERATIONS",
    "self_play_train_epochs": "SELF_PLAY_TRAIN_EPOCHS",
    "self_play_eval_games": "SELF_PLAY_EVAL_GAMES",
    "self_play_opponent_pool_size": "SELF_PLAY_OPPONENT_POOL_SIZE",
    "self_play_use_beam": "SELF_PLAY_USE_BEAM",
    "self_play_output_path": "SELF_PLAY_OUTPUT_PATH",
    "self_play_checkpoint_dir": "SELF_PLAY_CHECKPOINT_DIR",
    "self_play_train_after_collect": "SELF_PLAY_TRAIN_AFTER_COLLECT",
    "self_play_field_deck_dir": "SELF_PLAY_FIELD_DECK_DIR",
    "self_play_matchup_mode": "SELF_PLAY_MATCHUP_MODE",
    "self_play_target_rank": "SELF_PLAY_TARGET_RANK",
    "self_play_target_win_rate": "SELF_PLAY_TARGET_WIN_RATE",
    "self_play_plateau_patience": "SELF_PLAY_PLATEAU_PATIENCE",
    "self_play_workers": "SELF_PLAY_WORKERS",
    "self_play_baseline_dir": "SELF_PLAY_BASELINE_DIR",
    "self_play_baseline_win_rate": "SELF_PLAY_BASELINE_WIN_RATE",
    "self_play_baseline_archetype_decks_only": "SELF_PLAY_BASELINE_ARCHETYPE_DECKS_ONLY",
    "self_play_baseline_top_decks_per_archetype": "SELF_PLAY_BASELINE_TOP_DECKS_PER_ARCHETYPE",
    "self_play_agent_deck_dir": "SELF_PLAY_AGENT_DECK_DIR",
    "self_play_per_deck_checkpoint_dir": "SELF_PLAY_PER_DECK_CHECKPOINT_DIR",
    "self_play_train_window_games": "SELF_PLAY_TRAIN_WINDOW_GAMES",
    "self_play_trim_rollout_file": "SELF_PLAY_TRIM_ROLLOUT_FILE",
    "self_play_warmup_iterations": "SELF_PLAY_WARMUP_ITERATIONS",
    "self_play_warmup_lr_multiplier": "SELF_PLAY_WARMUP_LR_MULTIPLIER",
}


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in {
        "require_cabt_eval_data",
        "train_use_amp",
        "train_grad_checkpoint",
        "require_top_of_ladder_data",
    }:
        if isinstance(value, str):
            return value not in {"0", "false", "False", "no", "No"}
        return bool(value)
    if key in {"policy_weighting", "train_resume", "train_data_device", "train_tensor_cache"}:
        return str(value)
    if key == "train_tensor_cache_dir":
        return str(value)
    if key == "self_play_per_deck_checkpoint_dir" and value in {"", "none", "None", "null"}:
        return None
    if key == "self_play_train_window_games":
        if value is None or value == "":
            return None
        parsed = int(value)
        return None if parsed <= 0 else parsed
    if key == "dataset_games":
        if value is None or value == "":
            return None
        parsed = int(value)
        return None if parsed <= 0 else parsed
    if key in {
        "self_play_use_beam",
        "self_play_train_after_collect",
        "self_play_baseline_archetype_decks_only",
        "self_play_trim_rollout_file",
        "require_complete_games",
        "require_training_matchup_diversity",
    }:
        if isinstance(value, str):
            return value not in {"0", "false", "False", "no", "No"}
        return bool(value)
    if key in {
        "transition_classes",
        "state_hash_dim",
        "window_size",
        "card_vocab_size",
        "card_embed_dim",
        "d_model",
        "model_heads",
        "model_layers",
        "model_ff",
        "train_epochs",
        "early_stop_patience",
        "train_print_every",
        "batch_games",
        "tensor_build_workers",
        "beam_width",
        "beam_time_budget_ms",
        "beam_min_remaining_sec",
        "self_play_beam_width",
        "self_play_beam_time_budget_ms",
        "self_play_beam_max_search_steps",
        "self_play_beam_rollout_policy_width",
        "self_play_games",
        "self_play_iterations",
        "self_play_train_epochs",
        "self_play_eval_games",
        "self_play_opponent_pool_size",
        "self_play_target_rank",
        "self_play_plateau_patience",
        "self_play_workers",
        "self_play_warmup_iterations",
        "min_training_matchups",
        "min_training_deck_slugs",
        "train_checkpoint_every",
    }:
        return int(value)
    if key == "top_episode_percent":
        return float(value)
    if key == "training_deck_dirs":
        return list(value)
    if key in {
        "model_dropout",
        "learning_rate",
        "weight_decay",
        "loss_value_weight",
        "loss_policy_weight",
        "loss_dynamics_weight",
        "loss_entropy_weight",
        "loss_uncertainty_weight",
        "early_stop_min_delta",
        "value_win",
        "value_not_win",
        "value_timeout",
        "value_return_gamma",
        "value_shaping_alpha",
        "policy_awr_beta",
        "policy_awr_weight_max",
        "self_play_target_win_rate",
        "self_play_baseline_win_rate",
        "self_play_baseline_top_decks_per_archetype",
        "self_play_warmup_lr_multiplier",
        "min_top_of_ladder_fraction",
    }:
        return float(value)
    if key in {"fallback_rollout_data", "training_rollout_sources"}:
        return list(value)
    return value


def _resolve_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = default_user_config()
    for key, env_name in _ENV_MAP.items():
        if env_name in os.environ:
            settings[key] = _coerce_value(key, os.environ[env_name])
    if "REQUIRE_CABT_EVAL_DATA" in os.environ:
        settings["require_cabt_eval_data"] = os.environ["REQUIRE_CABT_EVAL_DATA"] != "0"
    if "BATCH_SIZE" in os.environ and "BATCH_GAMES" not in os.environ:
        settings["batch_games"] = _coerce_value("batch_games", os.environ["BATCH_SIZE"])
    if overrides:
        unknown = set(overrides) - OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"Unknown config overrides: {sorted(unknown)}")
        for key, value in overrides.items():
            settings[key] = _coerce_value(key, value)
    return settings


def resolve_self_play_train_window(settings: dict[str, Any]) -> int:
    """Games to keep for retrain/trim; defaults to games collected per iteration (--games)."""
    explicit = settings.get("self_play_train_window_games")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    return max(1, int(settings["self_play_games"]))


def build_config(root: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _resolve_settings(overrides)
    ensure_output_layout(root)
    d_model = settings["d_model"]
    model_ff = settings["model_ff"] if settings["model_ff"] is not None else d_model * 4
    fallback_paths = [root / path for path in settings["fallback_rollout_data"]]
    training_sources = [root / path for path in settings["training_rollout_sources"]]
    merged_path = root / settings["merged_rollout_data"]
    model_id = str(settings["model_id"])
    checkpoint = resolve_checkpoint_path(
        root,
        model_id=model_id,
        explicit=settings["model_output_path"],
    )
    data_candidates = list(dict.fromkeys([
        merged_path,
        *[path for path in training_sources if path != merged_path],
        root / settings["primary_rollout_data"],
        *fallback_paths,
    ]))
    train_window_games = resolve_self_play_train_window(settings)

    return {
        "root": root,
        "agent_deck_path": root / settings["agent_deck_path"],
        "data_candidates": data_candidates,
        "training_rollout_sources": training_sources,
        "merged_rollout_path": merged_path,
        "scraped_rollout_path": root / settings["scraped_rollout_data"],
        "multideck_rollout_path": root / settings["multideck_rollout_data"],
        "episodes_index_path": root / settings["episodes_index_path"],
        "top_episode_percent": settings["top_episode_percent"],
        "require_complete_games": settings["require_complete_games"],
        "require_training_matchup_diversity": settings["require_training_matchup_diversity"],
        "min_training_matchups": settings["min_training_matchups"],
        "min_training_deck_slugs": settings["min_training_deck_slugs"],
        "require_top_of_ladder_data": settings["require_top_of_ladder_data"],
        "min_top_of_ladder_fraction": settings["min_top_of_ladder_fraction"],
        "generated_path": root / settings["cabt_generated_path"],
        "competition_results_path": root / settings["competition_results_path"],
        "output_path": checkpoint,
        "report_path": report_path(root, model_id),
        "model_id": model_id,
        "output_layout": describe_layout(root),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "structured_feature_dim": STRUCTURED_FEATURE_DIM,
        "coarse_feature_dim": STRUCTURED_FEATURE_DIM,
        "require_cabt_eval_data": settings["require_cabt_eval_data"],
        "dataset_games": settings["dataset_games"],
        "transition_classes": settings["transition_classes"],
        "state_hash_dim": settings["state_hash_dim"],
        "window_size": settings["window_size"],
        "card_vocab_size": settings["card_vocab_size"],
        "card_embed_dim": settings["card_embed_dim"],
        "input_dim": total_feature_dim(state_hash_dim=settings["state_hash_dim"]),
        "tensor_build_workers": settings["tensor_build_workers"],
        "train_use_amp": settings["train_use_amp"],
        "train_grad_checkpoint": settings["train_grad_checkpoint"],
        "train_data_device": settings["train_data_device"],
        "train_tensor_cache": settings["train_tensor_cache"],
        "train_tensor_cache_dir": settings["train_tensor_cache_dir"],
        "objective": {
            "value_return_gamma": settings["value_return_gamma"],
            "value_shaping_alpha": settings["value_shaping_alpha"],
            "value_lambda": settings["value_lambda"],
            "value_mc_blend": settings["value_mc_blend"],
            "policy_weighting": settings["policy_weighting"],
            "policy_awr_beta": settings["policy_awr_beta"],
            "policy_awr_weight_max": settings["policy_awr_weight_max"],
            "policy_soft_topk": settings["policy_soft_topk"],
            "use_soft_search_policy": settings["objective_use_soft_search_policy"],
            "search_policy_kl_weight": settings["objective_search_policy_kl_weight"],
        },
        "model": {
            "d_model": d_model,
            "heads": settings["model_heads"],
            "layers": settings["model_layers"],
            "ff": model_ff,
            "dropout": settings["model_dropout"],
            "learning_rate": settings["learning_rate"],
            "encoder_learning_rate": settings["train_encoder_lr"],
            "head_learning_rate": settings["train_head_lr"],
            "weight_decay": settings["weight_decay"],
        },
        "loss": {
            "value": settings["loss_value_weight"],
            "policy": settings["loss_policy_weight"],
            "dynamics": settings["loss_dynamics_weight"],
            "entropy": settings["loss_entropy_weight"],
            "uncertainty": settings["loss_uncertainty_weight"],
        },
        "training": {
            "games": settings["dataset_games"],
            "epochs": settings["train_epochs"],
            "patience": settings["early_stop_patience"],
            "min_delta": settings["early_stop_min_delta"],
            "early_stop_metric": settings["early_stop_metric"],
            "print_every": settings["train_print_every"],
            "batch_games": settings["batch_games"],
            "checkpoint_every": settings["train_checkpoint_every"],
            "resume": settings["train_resume"],
        },
        "rewards": {
            "value_win": settings["value_win"],
            "value_not_win": settings["value_not_win"],
            "value_timeout": settings["value_timeout"],
        },
        "beam_search": {
            "width": settings["beam_width"],
            "time_budget_ms": settings["beam_time_budget_ms"],
            "min_remaining_sec": settings["beam_min_remaining_sec"],
        },
        "self_play": {
            "games_per_iteration": settings["self_play_games"],
            "iterations": settings["self_play_iterations"],
            "train_epochs": settings["self_play_train_epochs"],
            "eval_games": settings["self_play_eval_games"],
            "opponent_pool_size": settings["self_play_opponent_pool_size"],
            "use_beam": settings["self_play_use_beam"],
            "output_path": settings["self_play_output_path"],
            "checkpoint_dir": settings["self_play_checkpoint_dir"],
            "train_after_collect": settings["self_play_train_after_collect"],
            "field_deck_dir": settings["self_play_field_deck_dir"],
            "matchup_mode": settings["self_play_matchup_mode"],
            "target_rank": settings["self_play_target_rank"],
            "target_win_rate": settings["self_play_target_win_rate"],
            "plateau_patience": settings["self_play_plateau_patience"],
            "workers": settings["self_play_workers"],
            "baseline_dir": settings["self_play_baseline_dir"],
            "baseline_win_rate": settings["self_play_baseline_win_rate"],
            "baseline_archetype_decks_only": settings["self_play_baseline_archetype_decks_only"],
            "baseline_top_decks_per_archetype": settings["self_play_baseline_top_decks_per_archetype"],
            "agent_deck_dir": settings["self_play_agent_deck_dir"],
            "per_deck_checkpoint_dir": settings["self_play_per_deck_checkpoint_dir"],
            "train_window_games": train_window_games,
            "trim_rollout_file": settings["self_play_trim_rollout_file"],
            "warmup_iterations": settings["self_play_warmup_iterations"],
            "warmup_lr_multiplier": settings["self_play_warmup_lr_multiplier"],
            "beam_width": settings["self_play_beam_width"],
            "beam_time_budget_ms": settings["self_play_beam_time_budget_ms"],
            "beam_max_search_steps": settings["self_play_beam_max_search_steps"],
            "beam_rollout_policy_width": settings["self_play_beam_rollout_policy_width"],
            "search_determinizations": settings["search_determinizations"],
            "collection_inference_device": settings["collection_inference_device"],
            "opponent_latest_prob": settings["self_play_opponent_latest_prob"],
        },
    }


def resolve_generate_games(config: dict[str, Any]) -> int:
    """Games to simulate when generating rollouts (0 = skip)."""
    games = config.get("dataset_games")
    if games is not None:
        return max(0, int(games))
    return 0
