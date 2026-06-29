"""Self-play package.

Historically a single 2.4k-line module; now split into:
  - ``metrics``     — pure outcome + value-calibration math
  - ``rollout_io``  — rollout JSONL + manifest I/O
  - ``core``        — simulator collection, multiprocessing workers, eval, and the
                      baseline/curriculum/self-play orchestration loops

The multiprocessing worker globals and their init/task pairs are kept together in
``core`` because ``ProcessPoolExecutor`` initializers and ``global`` bindings must
share one module namespace. This ``__init__`` re-exports the full public surface so
``from poke_agent.self_play import ...`` keeps working for scripts and tests.
"""

from __future__ import annotations

# Patch target compatibility: tests monkeypatch ``poke_agent.self_play.random.random``.
import random  # noqa: F401

from poke_agent.self_play.metrics import (
    calibration_metrics_from_rows,
    record_seat_outcome as _record_seat_outcome,
    summarize_results,
    terminal_result as _terminal_result,
    value_calibration_metrics,
)

# Legacy private alias still imported by tests.
_calibration_metrics_from_rows = calibration_metrics_from_rows
from poke_agent.self_play.rollout_io import (
    count_rollout_games as _count_rollout_games,
    load_manifest,
    maybe_trim_rollout_file,
    rollout_buffer_overwrites,
    save_manifest,
    write_jsonl,
    write_rollout_buffer,
)
from poke_agent.self_play.core import (
    OpponentPool,
    SelfPlaySettings,
    _merge_eval_reports,
    _merge_fixed_opponent_reports,
    _resolve_agent_deck_pool,
    _resolve_baseline_resume,
    annotate_matchup_metadata,
    choose_pool_deck,
    collect_games_vs_baselines,
    collect_self_play_games,
    evaluate_agent_vs_field,
    evaluate_vs_baselines,
    probe_value_calibration,
    resolve_collection_device,
    resolve_matchup,
    resolve_self_play_workers,
    resolve_target_eval_matchup,
    run_baseline_iteration,
    run_baseline_phase_loop,
    run_curriculum_self_play,
    run_self_play_iteration,
    run_self_play_loop,
    self_play_settings_from_config,
    train_on_rollouts,
)

__all__ = [
    "OpponentPool",
    "SelfPlaySettings",
    "annotate_matchup_metadata",
    "calibration_metrics_from_rows",
    "choose_pool_deck",
    "collect_games_vs_baselines",
    "collect_self_play_games",
    "evaluate_agent_vs_field",
    "evaluate_vs_baselines",
    "load_manifest",
    "maybe_trim_rollout_file",
    "probe_value_calibration",
    "resolve_collection_device",
    "resolve_matchup",
    "resolve_self_play_workers",
    "resolve_target_eval_matchup",
    "rollout_buffer_overwrites",
    "run_baseline_iteration",
    "run_baseline_phase_loop",
    "run_curriculum_self_play",
    "run_self_play_iteration",
    "run_self_play_loop",
    "save_manifest",
    "self_play_settings_from_config",
    "summarize_results",
    "train_on_rollouts",
    "value_calibration_metrics",
    "write_jsonl",
    "write_rollout_buffer",
]
