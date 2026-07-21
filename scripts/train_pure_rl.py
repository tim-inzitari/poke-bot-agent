#!/usr/bin/env python
"""Single-lineage pure-RL loop: full-hardware collect → AWR train → held-out gate.

Modes:
  --mode core         deck-agnostic Stage A (default)
  --mode specialist   one explicitly named pinned archetype after warm-start
  --smoke             synthetic games (no CABT) for CI / canary wiring

Production collect saturates local CPU + dual-GPU leaves and optionally
additive whole-game farms (Elmo/bert) into the same shard stream. One AWR
trainee on the host — remotes are collect capacity, not a second trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("POKEBOT_BLACKWELL_STRATEGY_HEADS", "0")

from poke_bot import alakazam_heuristics, config, paths  # noqa: E402
from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT  # noqa: E402
from poke_bot.pure_rl.aborts import evaluate_aborts  # noqa: E402
from poke_bot.pure_rl.curriculum import (  # noqa: E402
    stage_for_iteration,
    stage_to_dict,
)
from poke_bot.pure_rl.dataset_bridge import (  # noqa: E402
    StreamingReplayCache,
    dataset_from_shard,
    validated_replay_cache_manifest,
)
from poke_bot.pure_rl.eval_public import (  # noqa: E402
    OFFICIAL_BASELINE_IDS,
    aggregate_heldout_wr,
    heldout_exploration_decision,
    heldout_goal_rank,
)
from poke_bot.pure_rl.hardware import full_hardware_profile  # noqa: E402
from poke_bot.pure_rl.metrics import IterationMetrics, metrics_to_dict  # noqa: E402
from poke_bot.pure_rl.artifact_retention import apply_artifact_retention  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    carry_learner_candidate,
    commit_rehearsal_receipt,
    recover_rehearsal,
    rehearsal_due,
    rehearsal_paths,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_profile import (  # noqa: E402
    build_pure_rl_model,
    count_params,
    model_config_dict,
    pure_rl_model_config,
    validate_param_budget,
)
from poke_bot.process_memory import close_mp_queue, release_process_heap  # noqa: E402
from poke_bot.pure_rl.shards import (  # noqa: E402
    CompactDecision,
    CompactGame,
    CompactShardWriter,
)
from poke_bot.train import (  # noqa: E402
    TrainConfig,
    belief_card_vocab_from_state,
    rl_train_step,
    supervised_rehearsal_step,
)

DEFAULT_REMOTE_ENDPOINTS = "192.168.1.143:8765,bert.local:8766"
LADDER_DECK_MIX_PATH = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
LADDER_DECK_REPRESENTATIVES_PATH = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=("core", "specialist"), default="core")
    p.add_argument(
        "--specialist-archetype",
        default=None,
        help=(
            "Exact pinned ladder deck ID for specialist mode. Required for "
            "--mode specialist; the trainer never falls back to Hammer-Pult."
        ),
    )
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=256)
    p.add_argument(
        "--train-epochs",
        type=int,
        default=int(os.environ.get("PURE_RL_TRAIN_EPOCHS", "2")),
        help=(
            "Fresh-data AWR passes per iteration. Two is the production default; "
            "treat this as a measured hyperparameter, not a fixed design rule."
        ),
    )
    p.add_argument(
        "--train-games-per-batch",
        type=int,
        default=int(os.environ.get("PURE_RL_TRAIN_GAMES_PER_BATCH", "48")),
        help="Whole-game accumulation cap for each RL optimizer micro-batch.",
    )
    p.add_argument(
        "--train-max-decisions-per-batch",
        type=int,
        default=int(
            os.environ.get("PURE_RL_TRAIN_MAX_DECISIONS_PER_BATCH", "4096")
        ),
        help="Decision-timestep cap per RL micro-batch; CUDA OOMs split safely.",
    )
    p.add_argument(
        "--train-warmup-max-decisions-per-batch",
        type=int,
        default=int(
            os.environ.get(
                "PURE_RL_TRAIN_WARMUP_MAX_DECISIONS_PER_BATCH", "0"
            )
        ),
        help=(
            "Optional decision cap used for the first N RL iterations. Set "
            "this together with --train-warmup-iterations; 0 disables the "
            "schedule. This is useful for denser early updates to auxiliary "
            "heads before returning to the steady-state throughput cap."
        ),
    )
    p.add_argument(
        "--train-warmup-iterations",
        type=int,
        default=int(os.environ.get("PURE_RL_TRAIN_WARMUP_ITERATIONS", "0")),
        help=(
            "Number of zero-based iterations that use the warmup decision "
            "cap; iteration N and later use --train-max-decisions-per-batch."
        ),
    )
    p.add_argument(
        "--train-device-resident",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_TRAIN_DEVICE_RESIDENT", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Pack the exact stateless replay window once into CUDA memory and "
            "train from GPU-resident indices; capacity failure falls back to "
            "the bounded host path."
        ),
    )
    p.add_argument(
        "--train-device-resident-min-free-gib",
        type=float,
        default=float(
            os.environ.get("PURE_RL_TRAIN_DEVICE_RESIDENT_MIN_FREE_GIB", "8")
        ),
        help="VRAM headroom retained while packing the resident replay corpus.",
    )
    p.add_argument(
        "--archetype-aux-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_ARCHETYPE_AUX_LOSS_WEIGHT", "0.05")),
        help="Opponent-archetype classification loss used by curriculum RL.",
    )
    p.add_argument(
        "--opp-hand-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_OPP_HAND_LOSS_WEIGHT", "0.05")),
        help="Masked opponent-hand belief loss; exact simulator labels only.",
    )
    p.add_argument(
        "--opp-remainder-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_OPP_REMAINDER_LOSS_WEIGHT", "0.05")),
        help="Masked hidden-remainder belief loss; exact simulator labels only.",
    )
    p.add_argument(
        "--lethal-threat-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_LETHAL_THREAT_LOSS_WEIGHT", "0.025")),
        help="Masked near-term prize-take/lethal trajectory loss.",
    )
    p.add_argument(
        "--prize-race-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_PRIZE_RACE_LOSS_WEIGHT", "0.025")),
        help="Masked public prize-race state regression loss.",
    )
    p.add_argument(
        "--alakazam-guide-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_ALAKAZAM_GUIDE_LOSS_WEIGHT", "0")),
        help=(
            "Small training-only confidence-weighted CE for the versioned "
            "Alakazam action guide. Valid only for that specialist; core "
            "stays exactly 0."
        ),
    )
    p.add_argument("--collect-temperature", type=float, default=1.0)
    p.add_argument(
        "--official-collect-frac",
        type=float,
        default=float(os.environ.get("PURE_RL_OFFICIAL_COLLECT_FRAC", "0")),
        help=(
            "Fraction of the public-training wave assigned to the four official "
            "baseline policies (specialist mode only). Formal evaluation still "
            "uses a separate greedy seed schedule; 0 keeps them evaluation-only."
        ),
    )
    p.add_argument(
        "--official-adaptive-targeting",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_OFFICIAL_ADAPTIVE_TARGETING", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Allocate the official-training quota from the latest exact "
            "heldout gaps, concentrating games on weaker opponents while "
            "retaining a minimum share for every official baseline."
        ),
    )
    p.add_argument(
        "--official-adaptive-min-share",
        type=float,
        default=float(
            os.environ.get("PURE_RL_OFFICIAL_ADAPTIVE_MIN_SHARE", "0.05")
        ),
        help="Minimum fraction of the official quota retained per opponent.",
    )
    p.add_argument(
        "--official-adaptive-gap-power",
        type=float,
        default=float(
            os.environ.get("PURE_RL_OFFICIAL_ADAPTIVE_GAP_POWER", "2.0")
        ),
        help="Exponent applied to each exact heldout gap before normalization.",
    )
    p.add_argument(
        "--official-exploit-opponents",
        default=os.environ.get("PURE_RL_OFFICIAL_EXPLOIT_OPPONENTS", ""),
        help=(
            "Comma-separated official baseline IDs whose training rows use a "
            "seat-paired lower-temperature mixture. Empty disables it."
        ),
    )
    p.add_argument(
        "--official-exploit-frac",
        type=float,
        default=float(os.environ.get("PURE_RL_OFFICIAL_EXPLOIT_FRAC", "0")),
        help=(
            "Fraction of each selected official opponent's training seat-pairs "
            "that use --official-exploit-temperature."
        ),
    )
    p.add_argument(
        "--official-exploit-temperature",
        type=float,
        default=float(
            os.environ.get("PURE_RL_OFFICIAL_EXPLOIT_TEMPERATURE", "0.35")
        ),
        help="Sharpened sampling temperature for the selected official rows.",
    )
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument(
        "--initial-learner-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional new-lineage learner warm start. The protected rollout "
            "champion remains --base-checkpoint until normal promotion passes."
        ),
    )
    p.add_argument(
        "--initial-heldout-evidence",
        type=Path,
        default=None,
        help=(
            "Optional exact, trusted eval_vs_baselines report for the initial "
            "learner. It anchors learner retention without counting as a "
            "terminal gate or a completed RL iteration."
        ),
    )
    p.add_argument(
        "--initial-replay-shard",
        type=Path,
        action="append",
        default=[],
        help="Prior immutable replay shard(s) used only in the first two-shard window.",
    )
    p.add_argument("--smoke", action="store_true", help="Synthetic loop, no CABT")
    p.add_argument("--smoke-games", type=int, default=8)
    p.add_argument(
        "--heldout-games",
        type=int,
        default=None,
        help=(
            "Compatibility assertion only. With --active-gate-contract the "
            "trainer derives this count from the contract and rejects a mismatch."
        ),
    )
    p.add_argument(
        "--active-gate-contract",
        type=Path,
        default=None,
        help=(
            "Authoritative active-gate program. Required for production "
            "specialist runs; roster, allocation, seats, and package digests "
            "are derived from it."
        ),
    )
    p.add_argument(
        "--measurement-decks",
        default=os.environ.get("PURE_RL_MEASUREMENT_DECKS", ""),
        help=(
            "Comma-separated candidate-side deck IDs used by promotion and "
            "formal heldout games. Empty keeps the full training deck pool."
        ),
    )
    p.add_argument(
        "--heldout-remotes",
        action="store_true",
        default=str(os.environ.get("PURE_RL_HELDOUT_REMOTES", "0")).lower()
        in ("1", "true", "yes", "on"),
        help="Use capability-verified additive remotes for the formal held-out gate.",
    )
    p.add_argument("--gate-wr", type=float, default=0.70)
    p.add_argument(
        "--heldout-per-opponent-floor",
        type=float,
        default=0.50,
        help="Minimum point win rate against every held-out opponent.",
    )
    p.add_argument("--promotion-games", type=int, default=80)
    p.add_argument("--promotion-workers", type=int, default=8)
    p.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.45,
        help="Candidate-vs-incumbent non-regression floor for the confidence bound.",
    )
    p.add_argument("--promotion-confidence", type=float, default=0.90)
    p.add_argument("--promotion-bootstrap-resamples", type=int, default=2000)
    p.add_argument(
        "--min-usable-game-frac",
        type=float,
        default=0.98,
        help="Fail the collection wave below this retained-trajectory fraction.",
    )
    p.add_argument(
        "--resume",
        choices=("auto", "never"),
        default="auto",
        help="Resume only from the append-only loop_state.json ledger.",
    )
    p.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="Optional assertion for the ledger's next iteration; never rewinds it.",
    )
    p.add_argument(
        "--allow-clean-boundary-design-migration",
        action="store_true",
        help=(
            "Permit one audited, append-only migration of operational batch/source "
            "fields, but only at a clean committed iteration boundary."
        ),
    )
    p.add_argument(
        "--boundary-design-migration-reason",
        default=None,
        help="Operator reason stored in the append-only design migration receipt.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game-timeout-s", type=int, default=600)
    p.add_argument(
        "--allow-single-gpu",
        action="store_true",
        help="Skip dual-GPU leaf requirement (CI / laptop)",
    )
    p.add_argument(
        "--remote-worker-endpoints",
        default=None,
        help=(
            "Comma-separated whole-game farms (host:port). "
            f"Default production: {DEFAULT_REMOTE_ENDPOINTS}. "
            "Pass empty string to disable."
        ),
    )
    p.add_argument(
        "--no-remote-workers",
        action="store_true",
        help="Disable remote whole-game farms even if endpoints are up",
    )
    p.add_argument(
        "--leaf-eval",
        choices=("gpu-server", "cpu"),
        default="gpu-server",
        help="Local leaf inference mode for host workers",
    )
    p.add_argument(
        "--multi-env-per-worker",
        type=int,
        default=None,
        help=(
            "In-process LibcgMultiEnv battles per OS worker (default: off / 1). "
            "Also: POKEBOT_MULTI_ENV=1 (→4) or POKEBOT_MULTI_ENV_PER_WORKER=N. "
            "Keeps WorkerPool as default until set."
        ),
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=None,
        help=(
            "Pure-RL leaf coalesce window (ms). Default 0 for ~1.6M policy "
            "(does not change Hope-large RR config default of 4). "
            "Env: PURE_RL_LEAF_COALESCE_MS (scoped; ignores RR LEAF_SERVER_COALESCE_MS)."
        ),
    )
    p.add_argument(
        "--expert-rehearsal-every",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_EVERY", "0")),
        help="Run one saved ladder-policy rehearsal before every Nth RL iteration (0=off).",
    )
    p.add_argument(
        "--expert-rehearsal-before-first",
        action="store_true",
        help="Run one expert complement pass before local iteration 0, then use normal cadence.",
    )
    p.add_argument(
        "--expert-rehearsal-force-before",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_FORCE_BEFORE", "-1")),
        help=(
            "Run one durable expert rehearsal before this exact iteration in "
            "addition to the normal cadence (-1=disabled)."
        ),
    )
    p.add_argument(
        "--expert-manifest",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_EXPERT_MANIFEST"])
            if os.environ.get("PURE_RL_EXPERT_MANIFEST")
            else None
        ),
        help="Feature manifest or atomic rolling-window pointer used for rehearsal.",
    )
    p.add_argument(
        "--expert-rehearsal-epochs",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_EPOCHS", "1")),
    )
    p.add_argument(
        "--expert-rehearsal-lr",
        type=float,
        default=float(os.environ.get("PURE_RL_EXPERT_REHEARSAL_LR", "2e-5")),
    )
    p.add_argument(
        "--expert-rehearsal-batch-size",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_BATCH_SIZE", "8192")),
    )
    p.add_argument(
        "--expert-min-decisions",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_MIN_DECISIONS", "5000000")),
    )
    p.add_argument(
        "--continuous-learner-min-wr",
        type=float,
        default=float(os.environ.get("PURE_RL_CONTINUOUS_LEARNER_MIN_WR", "0.35")),
        help="Carry rejected learner candidates unless head-to-head WR falls below this floor.",
    )
    p.add_argument(
        "--artifact-history-iterations",
        type=int,
        default=int(os.environ.get("PURE_RL_ARTIFACT_HISTORY_ITERATIONS", "5")),
        help="Retain this many completed iteration shards/checkpoints, plus protected identities.",
    )
    p.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=float(os.environ.get("PURE_RL_MIN_FREE_DISK_GB", "100")),
        help="Refuse to open a new large replay shard below this free-space floor.",
    )
    args = p.parse_args(argv)
    specialist = str(args.specialist_archetype or "").strip().lower()
    if args.mode == "specialist" and not specialist:
        p.error("--mode specialist requires --specialist-archetype")
    if args.mode == "core" and specialist:
        p.error("--specialist-archetype is valid only with --mode specialist")
    if not 0.0 <= float(args.official_collect_frac) <= 1.0:
        p.error("--official-collect-frac must be in [0, 1]")
    if args.mode != "specialist" and float(args.official_collect_frac) > 0.0:
        p.error("--official-collect-frac is valid only with --mode specialist")
    if bool(args.official_adaptive_targeting) and (
        args.mode != "specialist" or float(args.official_collect_frac) <= 0.0
    ):
        p.error(
            "--official-adaptive-targeting requires specialist mode and a "
            "positive --official-collect-frac"
        )
    adaptive_min_share = float(args.official_adaptive_min_share)
    if not 0.0 <= adaptive_min_share <= 1.0 / len(OFFICIAL_BASELINE_IDS):
        p.error(
            "--official-adaptive-min-share must be in [0, "
            f"{1.0 / len(OFFICIAL_BASELINE_IDS):.2f}]"
        )
    if not 0.0 < float(args.official_adaptive_gap_power) <= 4.0:
        p.error("--official-adaptive-gap-power must be in (0, 4]")
    exploit_frac = float(args.official_exploit_frac)
    if not 0.0 <= exploit_frac <= 1.0:
        p.error("--official-exploit-frac must be in [0, 1]")
    exploit_temperature = float(args.official_exploit_temperature)
    if not 0.0 < exploit_temperature <= 1.0:
        p.error("--official-exploit-temperature must be in (0, 1]")
    exploit_raw = [
        value.strip().lower()
        for value in str(args.official_exploit_opponents or "").split(",")
        if value.strip()
    ]
    if len(exploit_raw) != len(set(exploit_raw)):
        p.error("--official-exploit-opponents contains duplicates")
    unknown_exploit = sorted(set(exploit_raw) - set(OFFICIAL_BASELINE_IDS))
    if unknown_exploit:
        p.error(
            "--official-exploit-opponents contains unknown official IDs: "
            + ",".join(unknown_exploit)
        )
    exploit_opponents = tuple(
        opponent_id
        for opponent_id in OFFICIAL_BASELINE_IDS
        if opponent_id in set(exploit_raw)
    )
    if bool(exploit_opponents) != bool(exploit_frac > 0.0):
        p.error(
            "--official-exploit-opponents and a positive "
            "--official-exploit-frac must be enabled together"
        )
    if exploit_opponents and (
        args.mode != "specialist" or float(args.official_collect_frac) <= 0.0
    ):
        p.error(
            "official exploit collection requires specialist mode and a positive "
            "--official-collect-frac"
        )
    for option, value in (
        ("--archetype-aux-loss-weight", args.archetype_aux_loss_weight),
        ("--opp-hand-loss-weight", args.opp_hand_loss_weight),
        ("--opp-remainder-loss-weight", args.opp_remainder_loss_weight),
        ("--lethal-threat-loss-weight", args.lethal_threat_loss_weight),
        ("--prize-race-loss-weight", args.prize_race_loss_weight),
    ):
        if float(value) < 0.0:
            p.error(f"{option} cannot be negative")
    guide_weight = float(args.alakazam_guide_loss_weight)
    if guide_weight < 0.0:
        p.error("--alakazam-guide-loss-weight cannot be negative")
    if guide_weight > 0.0 and not (
        args.mode == "specialist" and specialist == "alakazam"
    ):
        p.error(
            "--alakazam-guide-loss-weight is valid only for "
            "--mode specialist --specialist-archetype alakazam"
        )
    if guide_weight > 0.0 and not alakazam_heuristics.enabled():
        p.error(
            "nonzero --alakazam-guide-loss-weight requires "
            "POKEBOT_ALAKAZAM_GUIDE_TARGETS=1"
        )
    if float(args.train_device_resident_min_free_gib) < 2.0:
        p.error("--train-device-resident-min-free-gib must be at least 2 GiB")
    warmup_cap = int(args.train_warmup_max_decisions_per_batch)
    warmup_iterations = int(args.train_warmup_iterations)
    if warmup_cap < 0 or warmup_iterations < 0:
        p.error("training warmup cap and iteration count cannot be negative")
    if (warmup_cap == 0) != (warmup_iterations == 0):
        p.error(
            "--train-warmup-max-decisions-per-batch and "
            "--train-warmup-iterations must be enabled together"
        )
    args.specialist_archetype = specialist or None
    args.official_exploit_opponents = exploit_opponents
    return args


def _scheduled_train_decision_cap(
    iteration: int,
    *,
    steady_cap: int,
    warmup_cap: int = 0,
    warmup_iterations: int = 0,
) -> tuple[int, str]:
    """Resolve the deterministic per-iteration optimizer decision cap."""
    iteration = int(iteration)
    steady_cap = int(steady_cap)
    warmup_cap = int(warmup_cap)
    warmup_iterations = int(warmup_iterations)
    if iteration < 0:
        raise ValueError("training iteration cannot be negative")
    if steady_cap <= 0:
        raise ValueError("steady training decision cap must be positive")
    if warmup_cap < 0 or warmup_iterations < 0:
        raise ValueError("training warmup values cannot be negative")
    if (warmup_cap == 0) != (warmup_iterations == 0):
        raise ValueError("training warmup cap and iteration count must be paired")
    if warmup_iterations and iteration < warmup_iterations:
        return warmup_cap, "head_focus_warmup"
    return steady_cap, "steady_state"


def _continuous_learner_carry_decision(
    *,
    candidate_safety_ok: bool,
    candidate_safety_reason: str,
    heldout_audit_ok: bool,
    promoted: bool,
) -> tuple[bool, str]:
    """Separate safe learner progress from protected-record selection."""
    if not bool(candidate_safety_ok):
        return False, str(candidate_safety_reason or "candidate_safety_failed")
    if not bool(heldout_audit_ok):
        return False, "heldout_contract_audit_failed"
    return (
        True,
        "promoted_safety_carry"
        if bool(promoted)
        else "continuous_learner_safety_carry",
    )


def _run_dir(run_name: str, *, smoke: bool = False) -> Path:
    # Smoke artifacts are intentionally outside the production lineage
    # namespace.  A canary may use the same human-readable run name, but it
    # must never occupy or replace production shards/checkpoints/markers.
    namespace = "pure_rl_smoke" if smoke else "pure_rl"
    d = paths.OUTPUTS_DIR / namespace / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shards").mkdir(exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "metrics").mkdir(exist_ok=True)
    (d / "eval").mkdir(exist_ok=True)
    (d / "commits").mkdir(exist_ok=True)
    (d / "rehearsals").mkdir(exist_ok=True)
    (d / "artifact_receipts").mkdir(exist_ok=True)
    return d


LOOP_STATE_VERSION = 2


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a mutable pointer/ledger atomically without a torn JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable JSON artifact and fail rather than overwrite it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(data)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}") from exc


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _source_snapshot(root: Optional[Path] = None) -> dict[str, Any]:
    """Capture enough source identity to reproduce a dirty-tree experiment."""
    source_root = Path(root or ROOT).resolve()

    def _git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=source_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def _git_bytes(*args: str) -> bytes:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=source_root,
                check=False,
                capture_output=True,
                timeout=30,
            )
        except Exception:
            return b""
        return proc.stdout if proc.returncode == 0 else b""

    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    diff_bytes = _git_bytes("diff", "--binary", "HEAD")
    untracked_raw = _git_bytes("ls-files", "--others", "--exclude-standard", "-z")
    untracked: list[dict[str, Any]] = []
    for raw_name in sorted(x for x in untracked_raw.split(b"\0") if x):
        name = os.fsdecode(raw_name)
        first_part = Path(name).parts[0] if Path(name).parts else ""
        if first_part in {"outputs", "data", ".pytest_cache"}:
            continue
        path = source_root / name
        if not path.is_file():
            continue
        untracked.append(
            {
                "path": name,
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )

    git_head = _git("rev-parse", "HEAD") or None
    tree_hash = hashlib.sha256()
    tree_hash.update((git_head or "no-git-head").encode("utf-8"))
    tree_hash.update(b"\0tracked-diff\0")
    tree_hash.update(diff_bytes)
    for row in untracked:
        tree_hash.update(b"\0untracked\0")
        tree_hash.update(str(row["path"]).encode("utf-8", errors="surrogateescape"))
        tree_hash.update(b"\0")
        tree_hash.update(str(row["sha256"]).encode("ascii"))

    # Some deployed copies are intentionally not Git worktrees.  In that
    # case, hash the executable Python/config source rather than silently
    # reporting the same empty identity for every build.
    fallback_files: list[dict[str, Any]] = []
    if git_head is None:
        candidates: list[Path] = []
        for dirname in ("poke_bot", "scripts"):
            base = source_root / dirname
            if base.is_dir():
                candidates.extend(x for x in base.rglob("*") if x.is_file())
        for name in ("requirements.txt", "pytest.ini"):
            path = source_root / name
            if path.is_file():
                candidates.append(path)
        for path in sorted(set(candidates)):
            if (
                "__pycache__" in path.parts
                or path.suffix in (".pyc", ".log")
                or path.name.startswith(".")
            ):
                continue
            rel = path.relative_to(source_root).as_posix()
            digest = _sha256_file(path)
            fallback_files.append(
                {"path": rel, "size": int(path.stat().st_size), "sha256": digest}
            )
            tree_hash.update(b"\0source-file\0")
            tree_hash.update(rel.encode("utf-8"))
            tree_hash.update(b"\0")
            tree_hash.update(digest.encode("ascii"))

    return {
        "git_head": git_head,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": f"sha256:{hashlib.sha256(diff_bytes).hexdigest()}",
        "untracked_files": untracked,
        "fallback_source_files": fallback_files,
        "source_tree_sha256": f"sha256:{tree_hash.hexdigest()}",
        "python": sys.version,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _canonical_digest(payload: Any) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _path_content_identity(path: Path) -> dict[str, Any]:
    """Content identity for a file or an installed baseline directory."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return {
            "kind": "file",
            "path": str(path),
            "size": int(path.stat().st_size),
            "digest": _sha256_file(path),
        }
    if not path.is_dir():
        raise FileNotFoundError(f"identity path does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for child in sorted(x for x in path.rglob("*") if x.is_file()):
        if "__pycache__" in child.parts or child.suffix in (".pyc", ".log"):
            continue
        rows.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size": int(child.stat().st_size),
                "digest": _sha256_file(child),
            }
        )
    return {
        "kind": "directory",
        "path": str(path),
        "files": rows,
        "digest": _canonical_digest(rows),
    }


def _load_initial_heldout_evidence(
    path: Path,
    *,
    checkpoint: CheckpointIdentity,
    heldout_games: int,
) -> dict[str, Any]:
    """Validate an exact external audit used only as the new-lineage anchor."""
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = set(OFFICIAL_BASELINE_IDS)
    expected_per_opponent = int(heldout_games) // len(expected_ids)
    checkpoint_report = dict(report.get("checkpoint") or {})
    pooled = dict(report.get("pooled_formal") or {})
    deck_gate = dict(report.get("deck_agnostic_gate") or {})
    matchups = list(report.get("matchups") or [])
    by_id: dict[str, dict[str, Any]] = {}
    for raw in matchups:
        row = dict(raw or {})
        opponent_id = str(row.get("opponent_id") or "")
        if opponent_id in by_id:
            raise RuntimeError(
                f"initial heldout evidence repeats opponent {opponent_id!r}"
            )
        by_id[opponent_id] = row

    failures: list[str] = []
    if not bool(report.get("valid")):
        failures.append("report_not_valid")
    if not bool(report.get("trusted_formal")):
        failures.append("report_not_trusted_formal")
    if str(report.get("formal_mode") or "") != "policy":
        failures.append("formal_mode_not_policy")
    if list(report.get("failures") or []):
        failures.append("game_failures_present")
    if str(checkpoint_report.get("digest") or "") != checkpoint.digest:
        failures.append("checkpoint_digest_mismatch")
    if set(report.get("expected_opponents") or []) != expected_ids:
        failures.append("official_opponent_set_mismatch")
    if set(by_id) != expected_ids:
        failures.append("matchup_set_mismatch")
    if int(report.get("scheduled_jobs", -1)) != int(heldout_games):
        failures.append("scheduled_game_count_mismatch")
    if int(report.get("completed_jobs", -1)) != int(heldout_games):
        failures.append("completed_game_count_mismatch")
    if int(pooled.get("games", -1)) != int(heldout_games):
        failures.append("pooled_game_count_mismatch")
    if int(report.get("min_games_per_opponent", -1)) < expected_per_opponent:
        failures.append("minimum_per_opponent_too_small")
    if not bool(deck_gate.get("exact_deck_seat_balance")):
        failures.append("seat_balance_not_exact")

    per_opponent: dict[str, dict[str, float]] = {}
    for opponent_id in OFFICIAL_BASELINE_IDS:
        row = by_id.get(opponent_id, {})
        try:
            games = int(row.get("games", -1))
            wins = float(row.get("wins", -1.0))
            losses = float(row.get("losses", -1.0))
            draws = float(row.get("draws", -1.0))
        except (TypeError, ValueError):
            failures.append(f"invalid_matchup_numbers:{opponent_id}")
            continue
        if games != expected_per_opponent:
            failures.append(f"per_opponent_game_count_mismatch:{opponent_id}")
        if min(wins, losses, draws) < 0.0 or not math.isclose(
            wins + losses + draws, float(games), abs_tol=1e-6
        ):
            failures.append(f"invalid_matchup_outcomes:{opponent_id}")
        per_opponent[opponent_id] = {
            "games": games,
            "wins": wins + 0.5 * draws,
            "losses": losses,
            "draws": draws,
            "win_rate": (wins + 0.5 * draws) / max(games, 1),
        }

    if failures:
        raise RuntimeError(
            "initial heldout evidence failed closed: " + ",".join(failures)
        )
    return {
        "evidence_schema": 2,
        "iteration": -1,
        "checkpoint_digest": checkpoint.digest,
        "games": int(heldout_games),
        "win_rate": float(pooled["wr"]),
        "confidence_lower": float(pooled["interval_lower"]),
        "confidence_upper": float(pooled["interval_upper"]),
        "per_opponent": per_opponent,
        "audit": {
            "passed": True,
            "source": "trusted_external_new_lineage_anchor",
            "report": _path_content_identity(path),
            "terminal_gate_eligible": False,
        },
    }


def _deck_distribution_contract(
    decks: list[tuple[str, list[int]]],
    *,
    ladder_contract: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Bind collection to the exact resolved pool and current sampler."""
    n = len(decks)
    ladder_weights = dict((ladder_contract or {}).get("weights") or {})
    entries = []
    for name, cards in decks:
        normalized = [int(card) for card in cards]
        entries.append(
            {
                "name": str(name),
                "cards_sha256": _canonical_digest(normalized),
                "card_count": len(normalized),
                "sampling_weight": (
                    float(ladder_weights[str(name)])
                    if str(name) in ladder_weights
                    else (1.0 / n if n else 0.0)
                ),
            }
        )
    contract = {
        "schema": 1,
        "sampler": (
            "official_ladder_hamilton_sha256_schedule_v1"
            if ladder_contract is not None
            else "deterministic_uniform_round_robin_v1"
        ),
        "entries": entries,
        "digest": _canonical_digest(entries),
    }
    if ladder_contract is not None:
        contract["ladder"] = dict(ladder_contract)
    return contract


def _checkpoint_contract(path: Path, *, smoke: bool) -> dict[str, Any]:
    """Fail closed unless a checkpoint is the exact intended trusted profile."""
    from poke_bot import checkpoint, features

    path = Path(path).expanduser().resolve()
    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    extra = dict(payload.get("extra") or {})
    if extra.get("pure_rl") is not True:
        raise RuntimeError(f"checkpoint is not explicitly pure_rl: {path}")
    if bool(extra.get("smoke", False)) != bool(smoke):
        raise RuntimeError(
            f"checkpoint smoke/production profile mismatch at {path}: "
            f"checkpoint_smoke={bool(extra.get('smoke', False))} requested={smoke}"
        )
    trusted = checkpoint.assert_trusted_policy_checkpoint(path)
    actual = dict(payload.get("model_config") or {})
    expected_cfg = pure_rl_model_config(**({"dropout": 0.0} if smoke else {}))
    expected = model_config_dict(expected_cfg)
    if actual != expected:
        changed = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(
            f"checkpoint does not match exact intended pure-RL model profile at "
            f"{path}; changed_fields={changed}"
        )
    provenance = dict(payload.get("provenance") or {})
    feature_schema = provenance.get("feature_schema")
    if feature_schema != features.FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint feature schema mismatch at {path}: "
            f"checkpoint={feature_schema!r} runtime={features.FEATURE_SCHEMA_VERSION!r}"
        )
    state_dict = payload.get("model_state_dict") or {}
    # The state dict is the architecture authority. ``extra.param_count`` is
    # training-time telemetry and can be smaller when heads were temporarily
    # frozen while saving a checkpoint. Trusting it makes a valid checkpoint's
    # immutable model contract change across a restart.
    #
    # State dicts include the feature-schema scalar buffer; it is not a
    # trainable parameter. The production profile has no other persistent
    # non-parameter tensors, and every remaining value is shape-countable.
    parameter_count = sum(
        int(value.numel())
        for key, value in state_dict.items()
        if hasattr(value, "numel")
        and not str(key).endswith("_feature_schema_version")
    )
    return {
        "path": str(path),
        "digest": checkpoint.checkpoint_digest(path),
        "pure_rl": True,
        "smoke": bool(smoke),
        "trusted_policy": trusted,
        "model_profile": actual,
        "decision_context": str(actual.get("decision_context")),
        "max_context": int(actual.get("max_context", 0)),
        "feature_schema": feature_schema,
        "trainable_parameters": int(parameter_count),
        "rl_iteration": int(payload.get("rl_iteration", 0)),
    }


def _opponent_specs_contract(specs: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "id": str(spec.id),
                "source": str(spec.source),
                "content": _path_content_identity(Path(spec.path)),
            }
        )
    return rows


def _design_contract(
    *,
    args: argparse.Namespace,
    checkpoint_profile: dict[str, Any],
    lineage_base: dict[str, str],
    initial_learner: dict[str, str],
    initial_replay_shards: list[dict[str, Any]],
    source: dict[str, Any],
    decks: list[tuple[str, list[int]]],
    measurement_decks: list[tuple[str, list[int]]],
    ladder_contract: Optional[dict[str, Any]],
    endpoints: list[str],
    collect_specs: list[Any],
    official_specs: list[Any],
    heldout_specs: list[Any],
    multi_env_per_worker: int,
    leaf_coalesce_ms: float,
    initial_heldout_evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Immutable inputs whose drift would fork the experiment lineage."""
    hidden_engine_raw = os.environ.get("POKEBOT_LIBCG_PATH", "").strip()
    hidden_engine = (
        _path_content_identity(Path(hidden_engine_raw).expanduser().resolve())
        if hidden_engine_raw
        else None
    )
    return {
        "schema": 1,
        "run": {"mode": str(args.mode), "iterations": int(args.iterations)},
        "games": {
            "per_iteration": int(args.games_per_iter),
            "heldout": int(args.heldout_games),
            "heldout_additive_remotes": bool(args.heldout_remotes),
            "promotion": int(args.promotion_games),
            "minimum_usable_fraction": float(args.min_usable_game_frac),
        },
        "learner": {
            "epochs": int(args.train_epochs),
            "games_per_batch": int(args.train_games_per_batch),
            "max_decisions_per_batch": int(
                args.train_max_decisions_per_batch
            ),
            "warmup_max_decisions_per_batch": int(
                args.train_warmup_max_decisions_per_batch
            ),
            "warmup_iterations": int(args.train_warmup_iterations),
            "device_resident": bool(args.train_device_resident),
            "device_resident_min_free_gib": float(
                args.train_device_resident_min_free_gib
            ),
            "seed": int(args.seed),
            "continuous": True,
            "carry_min_head_to_head_wr": float(args.continuous_learner_min_wr),
            "archetype_aux_loss_weight": float(
                args.archetype_aux_loss_weight
            ),
            "opp_hand_loss_weight": float(args.opp_hand_loss_weight),
            "opp_remainder_loss_weight": float(
                args.opp_remainder_loss_weight
            ),
            "lethal_threat_loss_weight": float(
                args.lethal_threat_loss_weight
            ),
            "prize_race_loss_weight": float(args.prize_race_loss_weight),
            "alakazam_guide_loss_weight": float(
                args.alakazam_guide_loss_weight
            ),
            "alakazam_guide_targets_enabled": bool(alakazam_heuristics.enabled()),
            "profile": dict(checkpoint_profile["model_profile"]),
            "trainable_parameters": int(
                checkpoint_profile["trainable_parameters"]
            ),
            "decision_context": str(checkpoint_profile["decision_context"]),
            "max_context": int(checkpoint_profile["max_context"]),
            "feature_schema": checkpoint_profile["feature_schema"],
            "initial_checkpoint": dict(initial_learner),
            "initial_heldout_evidence": (
                dict(initial_heldout_evidence)
                if initial_heldout_evidence is not None
                else None
            ),
            "initial_replay_shards": list(initial_replay_shards),
        },
        "expert_rehearsal": {
            "every_iterations": int(args.expert_rehearsal_every),
            "before_first_iteration": bool(args.expert_rehearsal_before_first),
            "epochs": int(args.expert_rehearsal_epochs),
            "learning_rate": float(args.expert_rehearsal_lr),
            "requested_batch_size": int(args.expert_rehearsal_batch_size),
            "minimum_decisions": int(args.expert_min_decisions),
            # The pointer is mutable by design; every actual manifest digest is
            # frozen in a per-rehearsal receipt instead of this lineage contract.
            "rolling_manifest_pointer": (
                str(Path(args.expert_manifest).expanduser().resolve())
                if args.expert_manifest is not None
                else None
            ),
        },
        "artifact_retention": {
            "history_iterations": int(args.artifact_history_iterations),
            "minimum_free_disk_gb": float(args.min_free_disk_gb),
            "protected_identities": (
                "champion,heldout_champion,learner,lineage_base,opponent_pool"
            ),
        },
        "gates": {
            "heldout_wr": float(args.gate_wr),
            "heldout_per_opponent_floor": float(args.heldout_per_opponent_floor),
            "active_contract": (
                _path_content_identity(
                    Path(args.active_gate_contract).expanduser().resolve()
                )
                if args.active_gate_contract is not None
                else None
            ),
            "promotion_threshold": float(args.promotion_threshold),
            "promotion_confidence": float(args.promotion_confidence),
            "promotion_bootstrap_resamples": int(args.promotion_bootstrap_resamples),
        },
        "collection": {
            "behavior_policy": (
                "continuous_learner_after_h2h_safety_and_exact_audit_v2"
            ),
            "temperature": float(args.collect_temperature),
            "game_timeout_s": int(args.game_timeout_s),
            "self_play_fraction": float(config.PURE_RL.self_play_frac),
            "official_target_fraction_of_public": float(
                args.official_collect_frac
            ),
            "official_targeting": {
                "strategy": (
                    "latest_exact_heldout_gap_v1"
                    if bool(args.official_adaptive_targeting)
                    else "uniform_v1"
                ),
                "minimum_share_per_opponent": float(
                    args.official_adaptive_min_share
                ),
                "gap_power": float(args.official_adaptive_gap_power),
                "target_win_rate": float(args.heldout_per_opponent_floor),
                "formal_eval_disjoint": True,
            },
            "official_exploit": {
                "opponents": list(args.official_exploit_opponents),
                "fraction_per_selected_opponent": float(
                    args.official_exploit_frac
                ),
                "temperature": float(args.official_exploit_temperature),
                "pairing": "consecutive_occurrence_seat_pairs_v1",
            },
            "replay_window_shards": int(config.PURE_RL.replay_window_shards),
            "leaf_eval": str(args.leaf_eval),
            "multi_env_per_worker": int(multi_env_per_worker),
            "leaf_coalesce_ms": float(leaf_coalesce_ms),
            "auxiliary_targets": {
                "archetype": "sequence_opponent_archetype_when_registered",
                "belief": "private_training_engine_exact_same_state_only",
                "lethal": "posthoc_played_line_prize_take_horizon",
                "prize_race": "public_observation_prize_counts",
                "missing_targets_masked": True,
                "hidden_engine": hidden_engine,
            },
        },
        "remotes": {
            "disabled": bool(args.no_remote_workers),
            "endpoints": list(endpoints),
        },
        "deck_distribution": _deck_distribution_contract(
            decks, ladder_contract=ladder_contract
        ),
        "measurement_deck_distribution": _deck_distribution_contract(
            measurement_decks, ladder_contract=None
        ),
        "opponents": {
            "collect": _opponent_specs_contract(collect_specs),
            "heldout": _opponent_specs_contract(heldout_specs),
            "official_target_training": (
                _opponent_specs_contract(official_specs)
                if float(args.official_collect_frac) > 0.0
                else []
            ),
        },
        "lineage_base": dict(lineage_base),
        "source": {
            "git_head": source.get("git_head"),
            "source_tree_sha256": source.get("source_tree_sha256"),
        },
    }


def _design_fingerprint(contract: dict[str, Any]) -> str:
    return _canonical_digest(contract)


def _validate_design_fingerprint(
    *, state: dict[str, Any], manifest: dict[str, Any], current: dict[str, Any]
) -> str:
    current_digest = _design_fingerprint(current)
    state_digest = str(state.get("design_fingerprint") or "")
    manifest_digest = str(manifest.get("design_fingerprint") or "")
    stored_contract = manifest.get("design_contract")
    if not state_digest or not manifest_digest or not isinstance(stored_contract, dict):
        raise RuntimeError("resume lineage is missing its immutable design fingerprint")
    if state_digest != manifest_digest:
        raise RuntimeError(
            "resume design fingerprint disagrees between ledger and manifest: "
            f"ledger={state_digest} manifest={manifest_digest}"
        )
    if _design_fingerprint(stored_contract) != manifest_digest:
        raise RuntimeError("immutable manifest design contract digest is corrupt")
    if current_digest != manifest_digest:
        changed = sorted(
            key
            for key in set(current) | set(stored_contract)
            if current.get(key) != stored_contract.get(key)
        )
        raise RuntimeError(
            "resume design drift detected; use a new run name. "
            f"changed_sections={changed} stored={manifest_digest} current={current_digest}"
        )
    return current_digest


_BOUNDARY_MIGRATABLE_DESIGN_PATHS = frozenset(
    {
        "learner.games_per_batch",
        "learner.max_decisions_per_batch",
        "learner.warmup_max_decisions_per_batch",
        "learner.warmup_iterations",
        "games.per_iteration",
        "games.heldout",
        "gates.heldout_wr",
        "gates.heldout_per_opponent_floor",
        "gates.active_contract",
        "opponents.collect",
        "opponents.heldout",
        "expert_rehearsal.rolling_manifest_pointer",
        "collection.behavior_policy",
        "collection.official_exploit",
        "collection.official_targeting",
        "measurement_deck_distribution",
    }
)


def _changed_design_paths(before: Any, after: Any, prefix: str = "") -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[str] = set()
        for key in set(before) | set(after):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.add(child)
            else:
                changed.update(_changed_design_paths(before[key], after[key], child))
        return changed
    return set() if before == after else {prefix or "<root>"}


def _load_design_migration_chain(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Verify the immutable manifest plus every append-only boundary migration."""
    contract = manifest.get("design_contract")
    digest = str(manifest.get("design_fingerprint") or "")
    if not digest or not isinstance(contract, dict):
        raise RuntimeError("resume lineage is missing its immutable design fingerprint")
    if _design_fingerprint(contract) != digest:
        raise RuntimeError("immutable manifest design contract digest is corrupt")

    receipts: list[dict[str, Any]] = []
    migration_dir = Path(run_dir) / "design_migrations"
    for path in sorted(migration_dir.glob("migration_*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if int(receipt.get("schema", -1)) != 1:
            raise RuntimeError(f"unsupported design migration receipt: {path}")
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        if not isinstance(previous, dict) or not isinstance(current, dict):
            raise RuntimeError(f"malformed design migration receipt: {path}")
        previous_digest = str(receipt.get("previous_fingerprint") or "")
        current_digest = str(receipt.get("current_fingerprint") or "")
        if (
            previous_digest != digest
            or _design_fingerprint(previous) != previous_digest
            or previous != contract
            or _design_fingerprint(current) != current_digest
        ):
            raise RuntimeError(f"broken design migration chain at {path}")
        contract = current
        digest = current_digest
        receipts.append(receipt)
    return contract, digest, receipts


def _validate_or_migrate_design_fingerprint(
    *,
    run_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    current: dict[str, Any],
    allow_clean_boundary_migration: bool,
    migration_reason: Optional[str],
) -> str:
    """Validate lineage, or append one tightly-scoped clean-boundary migration."""
    stored, stored_digest, receipts = _load_design_migration_chain(run_dir, manifest)
    state_digest = str(state.get("design_fingerprint") or "")

    # A receipt is written before the mutable ledger. Recover that one safe crash
    # window without mutating the original manifest or losing the committed state.
    if state_digest != stored_digest:
        latest = receipts[-1] if receipts else None
        if (
            latest is not None
            and state_digest == str(latest.get("previous_fingerprint") or "")
            and int(state.get("next_iteration", -1))
            == int(latest.get("boundary_next_iteration", -2))
        ):
            state["design_fingerprint"] = stored_digest
            state.setdefault("design_migration_history", []).append(
                {
                    "receipt": str(latest.get("receipt") or ""),
                    "fingerprint": stored_digest,
                    "recovered": True,
                }
            )
            _atomic_json(Path(run_dir) / "loop_state.json", state)
            state_digest = stored_digest
        else:
            raise RuntimeError(
                "resume design fingerprint disagrees with effective migration chain: "
                f"ledger={state_digest} effective={stored_digest}"
            )

    current_digest = _design_fingerprint(current)
    if current_digest == stored_digest:
        return current_digest
    changed = sorted(_changed_design_paths(stored, current))
    if not allow_clean_boundary_migration:
        raise RuntimeError(
            "resume design drift detected; use a new run name or an explicit clean-"
            "boundary migration. "
            f"changed_paths={changed} stored={stored_digest} current={current_digest}"
        )
    if not migration_reason or not str(migration_reason).strip():
        raise RuntimeError("clean-boundary design migration requires an operator reason")
    disallowed = [
        path
        for path in changed
        if not any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in _BOUNDARY_MIGRATABLE_DESIGN_PATHS
        )
        and not path.startswith("source.")
    ]
    if disallowed:
        raise RuntimeError(
            "clean-boundary design migration changes non-operational fields: "
            f"{disallowed}"
        )
    next_iteration = int(state.get("next_iteration", -1))
    last_completed = int(state.get("last_completed_iteration", -2))
    if next_iteration <= 0 or last_completed != next_iteration - 1:
        raise RuntimeError(
            "design migration requires a completed N+1 boundary: "
            f"last={last_completed} next={next_iteration}"
        )
    commit_path = Path(run_dir) / "commits" / f"iter_{last_completed:05d}.json"
    next_artifacts = _iteration_artifact_paths(run_dir, next_iteration)
    preserved_collection = _verified_completed_collection_receipt(
        run_dir, state, stored
    )
    expected_shard = (
        Path(run_dir) / "shards" / f"iter_{next_iteration:05d}.jsonl"
    ).resolve()
    artifacts_are_preserved_collection = bool(
        preserved_collection is not None
        and {path.resolve() for path in next_artifacts} == {expected_shard}
    )
    if (
        not commit_path.is_file()
        or (next_artifacts and not artifacts_are_preserved_collection)
    ):
        raise RuntimeError(
            "design migration requires a clean boundary or one verified completed "
            "collection shard with no train/eval artifacts"
        )
    committed = json.loads(commit_path.read_text(encoding="utf-8"))
    commit_digest = str(committed.get("design_fingerprint") or "")
    same_boundary_digests = {stored_digest}
    for prior_receipt in receipts:
        if int(prior_receipt.get("boundary_next_iteration", -1)) == next_iteration:
            same_boundary_digests.add(
                str(prior_receipt.get("previous_fingerprint") or "")
            )
            same_boundary_digests.add(
                str(prior_receipt.get("current_fingerprint") or "")
            )
    if (
        int(committed.get("next_iteration", -1)) != next_iteration
        or commit_digest not in same_boundary_digests
    ):
        raise RuntimeError("boundary commit does not match the effective design lineage")
    for field in ("games_per_batch", "max_decisions_per_batch"):
        if int(current["learner"][field]) <= 0:
            raise RuntimeError(f"migrated learner.{field} must remain positive")
    if (
        isinstance(current.get("games"), dict)
        and "per_iteration" in current["games"]
        and int(current["games"]["per_iteration"]) <= 0
    ):
        raise RuntimeError("migrated games.per_iteration must remain positive")
    try:
        _scheduled_train_decision_cap(
            next_iteration,
            steady_cap=int(current["learner"]["max_decisions_per_batch"]),
            warmup_cap=int(
                current["learner"].get("warmup_max_decisions_per_batch", 0)
            ),
            warmup_iterations=int(
                current["learner"].get("warmup_iterations", 0)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid migrated learner batch schedule: {exc}") from exc

    migration_dir = Path(run_dir) / "design_migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    index = len(receipts) + 1
    receipt_path = migration_dir / f"migration_{index:04d}.json"
    receipt = {
        "schema": 1,
        "receipt": str(receipt_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": str(migration_reason).strip(),
        "boundary_next_iteration": next_iteration,
        "last_completed_iteration": last_completed,
        "changed_paths": changed,
        "previous_fingerprint": stored_digest,
        "current_fingerprint": current_digest,
        "previous_contract": stored,
        "current_contract": current,
        "preserved_completed_collection": (
            {
                "receipt": str(preserved_collection.get("receipt_path") or ""),
                "iteration": int(preserved_collection["iteration"]),
                "checkpoint_digest": str(
                    preserved_collection["checkpoint_digest"]
                ),
            }
            if artifacts_are_preserved_collection
            else None
        ),
    }
    _write_json_exclusive(receipt_path, receipt)
    state["design_fingerprint"] = current_digest
    state.setdefault("design_migration_history", []).append(
        {
            "receipt": str(receipt_path),
            "fingerprint": current_digest,
            "boundary_next_iteration": next_iteration,
            "reason": str(migration_reason).strip(),
        }
    )
    _atomic_json(Path(run_dir) / "loop_state.json", state)
    print(
        f"[pure_rl] DESIGN_MIGRATION boundary_next={next_iteration} "
        f"changed={changed} receipt={receipt_path}",
        flush=True,
    )
    return current_digest


def _iteration_artifact_paths(run_dir: Path, iteration: int) -> list[Path]:
    stem = f"iter_{int(iteration):05d}"
    paths_out = [
        run_dir / "shards" / f"{stem}.jsonl",
        run_dir / "checkpoints" / f"{stem}.pt",
        run_dir / "eval" / f"{stem}.json",
        run_dir / "metrics" / f"{stem}.json",
    ]
    for parent, pattern in (
        (run_dir / "checkpoints", f"{stem}.pt.tmp.*"),
        (run_dir / "shards", f".{stem}*.tmp"),
    ):
        paths_out.extend(sorted(parent.glob(pattern)))
    latest = run_dir / "metrics" / "latest.json"
    if latest.is_file():
        try:
            if int(json.loads(latest.read_text(encoding="utf-8")).get("iteration")) == int(
                iteration
            ):
                paths_out.append(latest)
        except Exception:
            # A torn/malformed mutable pointer is also recovery debris.
            paths_out.append(latest)
    return [path for path in paths_out if path.exists()]


_COMPLETED_COLLECTION_SCHEMA = "poke_bot.completed_collection/v1"
def _collection_receipt_path(run_dir: Path, iteration: int) -> Path:
    return (
        Path(run_dir)
        / "collection_receipts"
        / f"iter_{int(iteration):05d}.json"
    )


def _completed_collection_digest(path: Path) -> str:
    """Hash the shard at every receipt trust boundary."""
    return _sha256_file(Path(path).resolve())


def _scan_completed_compact_shard(
    path: Path,
    *,
    expected_checkpoint_digest: Optional[str] = None,
) -> dict[str, Any]:
    """Validate/count one shard with bounded memory while hashing it.

    Legacy recovery has lost the in-memory dispatch audit.  Reconstruct its
    strongest invariants directly from every durable row: episode identities
    are unique and every trajectory was generated by the exact learner at the
    ledger boundary.  A merely parseable mixture of stale weights is not a
    recoverable completed collection.
    """
    path = Path(path)
    digest = hashlib.sha256()
    games = 0
    decisions = 0
    episode_ids: set[str] = set()
    observed_checkpoint_digests: set[str] = set()
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                raise RuntimeError(
                    f"completed collection has a partial final row: {path}"
                )
            digest.update(raw)
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise RuntimeError(f"compact shard row is not an object: {path}")
            episode_id = str(obj.get("episode_id") or "")
            if not episode_id:
                raise RuntimeError(
                    f"completed collection row has no episode_id: {path}"
                )
            if episode_id in episode_ids:
                raise RuntimeError(
                    "completed collection has a duplicate episode_id: "
                    f"{episode_id!r}"
                )
            episode_ids.add(episode_id)
            provenance = obj.get("target_provenance")
            if not isinstance(provenance, dict):
                raise RuntimeError(
                    f"completed collection row has no target provenance: {path}"
                )
            behavior_digest = str(
                provenance.get("behavior_checkpoint_digest") or ""
            )
            if not behavior_digest:
                raise RuntimeError(
                    "completed collection row has no behavior checkpoint digest: "
                    f"{episode_id!r}"
                )
            observed_checkpoint_digests.add(behavior_digest)
            if (
                expected_checkpoint_digest is not None
                and behavior_digest != str(expected_checkpoint_digest)
            ):
                raise RuntimeError(
                    "completed collection contains a stale/mixed behavior digest: "
                    f"episode={episode_id!r} got={behavior_digest!r} "
                    f"expected={expected_checkpoint_digest!r}"
                )
            rows = obj.get("decisions")
            if not isinstance(rows, list) or not rows:
                raise RuntimeError(
                    f"completed collection row has no decisions: {path}"
                )
            games += 1
            decisions += len(rows)
    if games <= 0 or decisions <= 0:
        raise RuntimeError(f"completed collection is empty: {path}")
    stat = path.stat()
    digest_value = f"sha256:{digest.hexdigest()}"
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest_value,
        "games": games,
        "decisions": decisions,
        "unique_episode_ids": len(episode_ids),
        "behavior_checkpoint_digests": sorted(observed_checkpoint_digests),
    }


def _completed_collection_contract(
    contract: dict[str, Any],
) -> tuple[int, float, int]:
    games = dict(contract.get("games") or {})
    learner = dict(contract.get("learner") or {})
    expected = int(games.get("per_iteration") or 0)
    minimum_fraction = float(games.get("minimum_usable_fraction") or 0.0)
    max_context = int(learner.get("max_context") or 0)
    if expected <= 0 or not 0.0 < minimum_fraction <= 1.0 or max_context <= 0:
        raise RuntimeError("invalid completed-collection design contract")
    return expected, minimum_fraction, max_context


def _verified_completed_collection_receipt(
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Verify a receipt against the immutable shard, cache, and ledger."""
    iteration = int(state.get("next_iteration", -1))
    receipt_path = _collection_receipt_path(run_dir, iteration)
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected, minimum_fraction, max_context = _completed_collection_contract(
            contract
        )
        shard_row = dict(receipt.get("shard") or {})
        shard = (
            Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
        ).resolve()
        stat = shard.stat()
        learner = dict(state.get("learner") or state.get("champion") or {})
        stats = dict(receipt.get("stats") or {})
        manifest = validated_replay_cache_manifest(
            shard,
            verify_info_set=False,
            max_context=max_context,
        )
        retained = int(stats.get("retained_source_games") or 0)
        if (
            receipt.get("schema") != _COMPLETED_COLLECTION_SCHEMA
            or int(receipt.get("iteration", -1)) != iteration
            or Path(str(shard_row.get("path") or "")).resolve() != shard
            or int(shard_row.get("size", -1)) != int(stat.st_size)
            or int(shard_row.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
            or not str(shard_row.get("sha256") or "").startswith("sha256:")
            or _completed_collection_digest(shard)
            != str(shard_row.get("sha256") or "")
            or int(receipt.get("requested_games", -1)) != expected
            or str(receipt.get("checkpoint_digest") or "")
            != str(learner.get("digest") or "")
            or retained < math.ceil(expected * minimum_fraction)
            or int(shard_row.get("games", -1)) <= 0
            or int(shard_row.get("decisions", -1)) <= 0
            or manifest is None
            or int(manifest.get("records", -1))
            != int(shard_row.get("games", -2))
            or int(manifest.get("sequences", -1))
            != int(shard_row.get("games", -2))
            or int(manifest.get("dropped", -1)) != 0
            or int(manifest.get("covered_bytes", -1)) != int(stat.st_size)
        ):
            return None
        return {**receipt, "receipt_path": str(receipt_path)}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _commit_completed_collection_receipt(
    *,
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
    iteration: int,
    shard: Path,
    checkpoint: Path,
    checkpoint_digest: str,
    stats: dict[str, Any],
    started_at: float,
    writer: Optional[CompactShardWriter] = None,
    recovery_derived: bool = False,
) -> dict[str, Any]:
    """Commit the collection/rehearsal transaction boundary exactly once."""
    expected, minimum_fraction, max_context = _completed_collection_contract(
        contract
    )
    if int(iteration) != int(state.get("next_iteration", -1)):
        raise RuntimeError("completed collection does not match ledger boundary")
    learner = dict(state.get("learner") or state.get("champion") or {})
    if str(checkpoint_digest) != str(learner.get("digest") or ""):
        raise RuntimeError("completed collection checkpoint disagrees with ledger")
    manifest = validated_replay_cache_manifest(
        shard,
        verify_info_set=False,
        max_context=max_context,
    )
    if manifest is None or int(manifest.get("dropped", -1)) != 0:
        raise RuntimeError("completed collection lacks a lossless replay cache")
    if recovery_derived:
        shard_row = _scan_completed_compact_shard(
            shard,
            expected_checkpoint_digest=str(checkpoint_digest),
        )
    else:
        stat = Path(shard).stat()
        if writer is None:
            raise RuntimeError("live collection receipt requires writer counters")
        shard_row = {
            "path": str(Path(shard).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": _sha256_file(shard),
            "games": int(writer.n_games),
            "decisions": int(writer.n_decisions),
        }
    if (
        int(manifest.get("records", -1)) != int(shard_row["games"])
        or int(manifest.get("sequences", -1)) != int(shard_row["games"])
        or int(manifest.get("covered_bytes", -1)) != int(shard_row["size"])
    ):
        raise RuntimeError("completed collection counters disagree with replay cache")
    normalized_stats = dict(stats)
    retained = int(normalized_stats.get("retained_source_games") or 0)
    if retained < math.ceil(expected * minimum_fraction):
        raise RuntimeError("completed collection is below the usable-game threshold")
    completed_at = float(
        normalized_stats.get("collect_completed_at")
        or (float(shard_row["mtime_ns"]) / 1_000_000_000.0)
    )
    elapsed = max(
        float(normalized_stats.get("collect_elapsed_sec") or 0.0),
        completed_at - float(started_at),
        1e-6,
    )
    normalized_stats["collect_elapsed_sec"] = elapsed
    payload = {
        "schema": _COMPLETED_COLLECTION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": int(iteration),
        "requested_games": expected,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_digest": str(checkpoint_digest),
        "design_fingerprint_at_collection": str(
            state.get("design_fingerprint") or ""
        ),
        "started_at": float(started_at),
        "completed_at": completed_at,
        "recovery_derived": bool(recovery_derived),
        "shard": shard_row,
        "replay_cache": {
            "manifest_path": manifest.get("manifest_path"),
            "records": int(manifest["records"]),
            "sequences": int(manifest["sequences"]),
            "dropped": int(manifest["dropped"]),
            "covered_bytes": int(manifest["covered_bytes"]),
            "signature": manifest.get("signature"),
        },
        "stats": normalized_stats,
    }
    receipt_path = _collection_receipt_path(run_dir, iteration)
    _write_json_exclusive(receipt_path, payload)
    print(
        f"[pure_rl] completed collection committed iter={iteration} "
        f"games={shard_row['games']} decisions={shard_row['decisions']} "
        f"receipt={receipt_path}",
        flush=True,
    )
    return {**payload, "receipt_path": str(receipt_path)}


def _ensure_recoverable_completed_collection(
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Recover the narrow crash window after collect and before train commit."""
    existing = _verified_completed_collection_receipt(run_dir, state, contract)
    if existing is not None:
        return existing
    iteration = int(state.get("next_iteration", -1))
    # A present-but-invalid immutable receipt must be quarantined together with
    # its shard.  Never overwrite it or infer that it described this retry.
    if _collection_receipt_path(run_dir, iteration).exists():
        return None
    shard = Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
    artifacts = _iteration_artifact_paths(run_dir, iteration)
    if not shard.is_file() or {path.resolve() for path in artifacts} != {
        shard.resolve()
    }:
        return None
    expected, minimum_fraction, max_context = _completed_collection_contract(
        contract
    )
    runtime_path = Path(run_dir) / "iteration_runtime.json"
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    learner = dict(state.get("learner") or state.get("champion") or {})
    manifest = validated_replay_cache_manifest(
        shard,
        verify_info_set=False,
        max_context=max_context,
    )
    # With the original in-memory stats gone, require the strongest exact
    # case. A merely 98%-usable partial wave is retried instead of guessed.
    if (
        str(runtime.get("phase") or "") != "collect"
        or int(runtime.get("iteration", -1)) != iteration
        or str(runtime.get("checkpoint_digest") or "")
        != str(learner.get("digest") or "")
        or manifest is None
        or int(manifest.get("records", -1)) != expected
        or int(manifest.get("sequences", -1)) != expected
        or int(manifest.get("dropped", -1)) != 0
    ):
        return None
    started_at = float(runtime.get("started_at") or shard.stat().st_mtime)
    elapsed = max(shard.stat().st_mtime - started_at, 1e-6)
    stats = {
        "ok": expected,
        "with_record": expected,
        "requested_games": expected,
        "retained_source_games": expected,
        "retained_trajectories": expected,
        "usable_game_fraction": 1.0,
        "collect_elapsed_sec": elapsed,
        "claimed_games_per_sec": expected / elapsed,
        "valid_source_games_per_sec": expected / elapsed,
        "trajectory_games_per_sec": expected / elapsed,
        "recovered_completed_collection": True,
        "minimum_usable_fraction": minimum_fraction,
    }
    try:
        return _commit_completed_collection_receipt(
            run_dir=run_dir,
            state=state,
            contract=contract,
            iteration=iteration,
            shard=shard,
            checkpoint=Path(str(learner.get("path") or "")),
            checkpoint_digest=str(learner.get("digest") or ""),
            stats=stats,
            started_at=started_at,
            recovery_derived=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            "[pure_rl] completed collection recovery rejected; "
            f"will quarantine/recollect iter={iteration}: {exc}",
            flush=True,
        )
        return None


def _completed_collection_bundle(
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    receipt = _verified_completed_collection_receipt(run_dir, state, contract)
    if receipt is None:
        return None
    shard_row = dict(receipt["shard"])
    elapsed = float((receipt.get("stats") or {}).get("collect_elapsed_sec") or 0.0)
    writer = CompactShardWriter.from_completed_shard(
        Path(str(shard_row["path"])),
        n_games=int(shard_row["games"]),
        n_decisions=int(shard_row["decisions"]),
        elapsed_sec=elapsed,
    )
    print(
        f"[pure_rl] resume completed collection iter={receipt['iteration']} "
        f"games={writer.n_games} decisions={writer.n_decisions} "
        "(skip recollection)",
        flush=True,
    )
    return {
        "iteration": int(receipt["iteration"]),
        "shard": Path(str(shard_row["path"])),
        "writer": writer,
        "stats": dict(receipt["stats"]),
        "checkpoint": Path(str(receipt["checkpoint"])),
        "digest": str(receipt["checkpoint_digest"]),
        "started_at": float(receipt["started_at"]),
        "recovered": True,
        "receipt": str(receipt["receipt_path"]),
    }


def _collection_behavior_identity(bundle: dict[str, Any]):
    """Return the digest-verified policy that produced this collection.

    The rollout champion and the continuous learner can intentionally differ.
    Training provenance must therefore come from the immutable collection
    receipt/bundle, never from the H2H incumbent selected later in the loop.
    """
    return _verified_checkpoint_identity(
        {
            "path": str(bundle.get("checkpoint") or ""),
            "digest": str(bundle.get("digest") or ""),
        }
    )


def _recover_interrupted_iteration(
    run_dir: Path, state: dict[str, Any]
) -> Optional[Path]:
    """Transactionally quarantine an uncommitted iteration so it can retry."""
    iteration = int(state.get("next_iteration", 0))
    commit = run_dir / "commits" / f"iter_{iteration:05d}.json"
    if commit.exists():
        raise RuntimeError(
            f"recovery called before committed iteration {iteration} was replayed"
        )
    artifacts = _iteration_artifact_paths(run_dir, iteration)
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.is_file():
        immutable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        effective_contract, _effective_digest, _receipts = (
            _load_design_migration_chain(run_dir, immutable_manifest)
        )
        completed = _verified_completed_collection_receipt(
            run_dir, state, effective_contract
        )
        expected_shard = (
            Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
        ).resolve()
        if completed is not None and {
            path.resolve() for path in artifacts
        } == {expected_shard}:
            print(
                f"[pure_rl] preserve receipt-verified completed collection "
                f"iter={iteration}; resume at rehearsal/train",
                flush=True,
            )
            return None
    receipt_path = _collection_receipt_path(run_dir, iteration)
    if receipt_path.exists() and receipt_path not in artifacts:
        artifacts.append(receipt_path)
    qroot = run_dir / "quarantine" / f"iter_{iteration:05d}"

    # Resume a quarantine transaction if the process died while moving files.
    in_progress: Optional[Path] = None
    if qroot.is_dir():
        for attempt in sorted(qroot.glob("attempt_*")):
            if (attempt / "plan.json").is_file() and not (
                attempt / "failure.json"
            ).is_file():
                in_progress = attempt
                break
    if in_progress is not None:
        plan = json.loads((in_progress / "plan.json").read_text(encoding="utf-8"))
    elif not artifacts:
        return None
    else:
        qroot.mkdir(parents=True, exist_ok=True)
        attempt_i = 1
        while (qroot / f"attempt_{attempt_i:04d}").exists():
            attempt_i += 1
        in_progress = qroot / f"attempt_{attempt_i:04d}"
        in_progress.mkdir()
        plan_rows = []
        for source in artifacts:
            rel = source.relative_to(run_dir)
            plan_rows.append(
                {
                    "source": str(source),
                    "relative_path": rel.as_posix(),
                    "destination": str(in_progress / rel),
                    "size": int(source.stat().st_size),
                    "digest": _sha256_file(source),
                }
            )
        plan = {
            "schema": 1,
            "iteration": iteration,
            "reason": "interrupted_before_append_only_commit",
            "ledger_next_iteration": iteration,
            "artifacts": plan_rows,
        }
        _write_json_exclusive(in_progress / "plan.json", plan)

    for row in plan.get("artifacts") or []:
        source = Path(row["source"])
        destination = Path(row["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256_file(destination) != str(row["digest"]):
                raise RuntimeError(
                    f"quarantine destination digest mismatch: {destination}"
                )
            if source.exists():
                raise RuntimeError(
                    f"quarantine has both source and destination: {source}"
                )
            continue
        if not source.exists():
            raise RuntimeError(
                f"quarantine lost both source and destination for {row['source']}"
            )
        if _sha256_file(source) != str(row["digest"]):
            raise RuntimeError(f"quarantine source changed during recovery: {source}")
        source.replace(destination)

    failure_path = in_progress / "failure.json"
    if not failure_path.exists():
        _write_json_exclusive(
            failure_path,
            {
                **plan,
                "quarantine_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Restore latest.json to the last committed metric when the interrupted
    # pointer was quarantined.  The pointer is mutable; iteration metrics are not.
    latest = run_dir / "metrics" / "latest.json"
    prior = run_dir / "metrics" / f"iter_{iteration - 1:05d}.json"
    if not latest.exists() and prior.is_file():
        _atomic_json(latest, json.loads(prior.read_text(encoding="utf-8")))
    return failure_path


def _load_loop_state(run_dir: Path) -> Optional[dict[str, Any]]:
    path = Path(run_dir) / "loop_state.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if int(state.get("version", -1)) != LOOP_STATE_VERSION:
        raise RuntimeError(
            f"unsupported pure-RL loop state version at {path}: "
            f"{state.get('version')!r}"
        )
    # A commit record is the transaction boundary. If the process died after
    # writing it but before advancing the mutable pointer, replay it exactly.
    while True:
        next_it = int(state.get("next_iteration", 0))
        commit_path = Path(run_dir) / "commits" / f"iter_{next_it:05d}.json"
        if not commit_path.is_file():
            break
        committed = json.loads(commit_path.read_text(encoding="utf-8"))
        if (
            int(committed.get("version", -1)) != LOOP_STATE_VERSION
            or int(committed.get("last_completed_iteration", -1)) != next_it
            or int(committed.get("next_iteration", -1)) != next_it + 1
            or committed.get("run_name") != state.get("run_name")
            or committed.get("mode") != state.get("mode")
        ):
            raise RuntimeError(f"invalid append-only commit record: {commit_path}")
        state = committed
        _atomic_json(Path(run_dir) / "loop_state.json", state)
    return state


def _validate_resume_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    state: dict[str, Any],
) -> tuple[Path, int]:
    """Resolve the immutable incumbent and monotonic next iteration."""
    from poke_bot.promotion import CheckpointIdentity

    if args.resume == "never":
        raise RuntimeError(
            f"run {args.run_name!r} already has loop_state.json; "
            "use a new run name or --resume auto"
        )
    if str(state.get("run_name")) != str(args.run_name):
        raise RuntimeError("loop state run_name does not match requested run")
    if str(state.get("mode")) != str(args.mode):
        raise RuntimeError("loop state mode does not match requested mode")
    champion = dict(state.get("champion") or {})
    ckpt = Path(str(champion.get("path") or "")).expanduser().resolve()
    identity = CheckpointIdentity.from_path(ckpt)
    expected = str(champion.get("digest") or "")
    if not expected or identity.digest != expected:
        raise RuntimeError(
            "resume champion digest mismatch: "
            f"ledger={expected!r} disk={identity.digest!r} path={ckpt}"
        )
    next_it = int(state.get("next_iteration", 0))
    if next_it < 0:
        raise RuntimeError(f"invalid next_iteration={next_it} in loop ledger")
    if args.start_iteration is not None and int(args.start_iteration) != next_it:
        raise RuntimeError(
            f"--start-iteration={args.start_iteration} would rewind/skip ledger "
            f"next_iteration={next_it}"
        )
    if args.base_checkpoint is not None:
        requested = Path(args.base_checkpoint).expanduser().resolve()
        if requested != ckpt:
            raise RuntimeError(
                "--base-checkpoint conflicts with resumed champion: "
                f"{requested} != {ckpt}"
            )
    return ckpt, next_it


def _verified_checkpoint_identity(entry: Any):
    """Return a path+digest identity and verify both against immutable bytes."""
    from poke_bot.promotion import CheckpointIdentity

    if isinstance(entry, CheckpointIdentity):
        expected_path, expected_digest = entry.path, entry.digest
    elif isinstance(entry, dict):
        expected_path = str(entry.get("path") or "")
        expected_digest = str(entry.get("digest") or "")
    else:
        raise RuntimeError(
            "opponent pool entries must be path+digest identities, got "
            f"{type(entry).__name__}"
        )
    if not expected_path or not expected_digest:
        raise RuntimeError(f"incomplete checkpoint identity: {entry!r}")
    actual = CheckpointIdentity.from_path(expected_path)
    if actual.digest != expected_digest:
        raise RuntimeError(
            "checkpoint identity digest mismatch: "
            f"path={actual.path} ledger={expected_digest} disk={actual.digest}"
        )
    return actual


def _verify_learner_lineage(
    result: dict[str, Any], *, candidate: Any, parent: Any
) -> None:
    reported_candidate_digest = str(result.get("candidate_digest") or "")
    if reported_candidate_digest != str(candidate.digest):
        raise RuntimeError(
            "learner returned a candidate digest that does not match disk: "
            f"reported={reported_candidate_digest!r} "
            f"disk={candidate.digest!r} path={candidate.path}"
        )
    reported_parent_digest = str(result.get("parent_digest") or "")
    if reported_parent_digest != str(parent.digest):
        raise RuntimeError(
            "learner returned the wrong parent lineage: "
            f"reported={reported_parent_digest!r} expected={parent.digest!r}"
        )


def _terminal_gate_payload(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    history = list(state.get("history") or [])
    if not history:
        return None
    row = dict(history[-1])
    gate = dict(row.get("stage_gate") or {})
    if not bool(gate.get("passed", False)):
        return None
    champion = _verified_checkpoint_identity(state.get("champion") or {})
    return {
        "iteration": int(row["iteration"]),
        "wr": float(gate["win_rate"]),
        "confidence_lower": float(gate["confidence_lower"]),
        "games": int(gate["games"]),
        "checkpoint": champion.path,
        "checkpoint_digest": champion.digest,
    }


def _reconciled_heldout_champion_evidence(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Upgrade legacy pooled-only evidence from its committed exact gate.

    v6 committed the full per-opponent gate in history but omitted it from the
    heldout-champion summary.  Reconstruct only when iteration, digest, audit,
    game count, and pooled WR all agree; otherwise leave the evidence untouched.
    """
    evidence = dict(state.get("heldout_champion_evidence") or {})
    if not evidence or isinstance(evidence.get("per_opponent"), dict):
        return evidence
    digest = str(evidence.get("checkpoint_digest") or "")
    try:
        evidence_iteration = int(evidence.get("iteration"))
        evidence_games = int(evidence.get("games"))
        evidence_wr = float(evidence.get("win_rate"))
    except (TypeError, ValueError):
        return evidence
    for raw_row in reversed(list(state.get("history") or [])):
        row = dict(raw_row or {})
        try:
            row_iteration = int(row.get("iteration", -1))
        except (TypeError, ValueError):
            continue
        if row_iteration != evidence_iteration:
            continue
        champion = dict(row.get("heldout_champion") or {})
        audit = dict(row.get("heldout_audit") or {})
        gate = dict(row.get("stage_gate") or {})
        per_opponent = gate.get("per_opponent")
        try:
            gate_games = int(gate.get("games", -1))
            gate_wr = float(gate.get("win_rate", -1.0))
        except (TypeError, ValueError):
            return evidence
        if (
            str(champion.get("digest") or "") != digest
            or not bool(audit.get("passed"))
            or not isinstance(per_opponent, dict)
            or any(oid not in per_opponent for oid in OFFICIAL_BASELINE_IDS)
            or gate_games != evidence_games
            or not math.isclose(
                gate_wr,
                evidence_wr,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return evidence
        upgraded = dict(evidence)
        upgraded["evidence_schema"] = 2
        upgraded["per_opponent"] = json.loads(json.dumps(per_opponent))
        upgraded["reconciled_from"] = "committed_history.stage_gate"
        return upgraded
    return evidence


def _ensure_terminal_gate_marker(
    run_dir: Path, state: dict[str, Any]
) -> Optional[Path]:
    """Recreate or validate the derived terminal marker idempotently."""
    payload = _terminal_gate_payload(state)
    if payload is None:
        return None
    marker = run_dir / (
        "CORE_GATE_PASSED"
        if str(state.get("mode")) == "core"
        else "SPECIALIST_GATE_PASSED"
    )
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid terminal gate marker: {marker}") from exc
        if existing != payload:
            raise RuntimeError(
                f"terminal gate marker disagrees with committed ledger: {marker}"
            )
    else:
        _write_json_exclusive(marker, payload)
    return marker


def _smoke_games(n: int, *, seed: int, archetype: str) -> list[CompactGame]:
    games: list[CompactGame] = []
    for i in range(n):
        games.append(
            CompactGame(
                episode_id=f"smoke-{seed}-{i}",
                seat=i % 2,
                archetype=archetype,
                opp_archetype="iono",
                deck=[1 + (i % 5)] * 60,
                value=1.0 if i % 3 else -1.0,
                decisions=[
                    CompactDecision(
                        env_step=0,
                        selected_index=i % 2,
                        n_options=2,
                        action=[i % 2],
                        observation={},
                    )
                ],
                target_provenance={"smoke": True, "pure_rl": True},
            )
        )
    return games


def _smoke_dataset(n: int, seed: int):
    from poke_bot import features
    from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage

    def sparse(words: int, offset: int = 0):
        sv = features.SparseVector()
        for i in range(words):
            sv.word_start()
            sv.add((offset + i) % 32, 1.0)
        return sv

    seqs = []
    for i in range(n):
        combos = [[0], [1]]
        dec = DecisionSample(
            board=sparse(features.NUM_BOARD_TOKENS, i),
            options=sparse(2, i + 3),
            action=[i % 2],
            action_combo_index=i % 2,
            action_combos=combos,
            env_step=0,
            action_token=sparse(1, i + 7),
            policy_stages=[
                PolicyStage(
                    options=sparse(2, i + 3),
                    action_combos=combos,
                    target_index=i % 2,
                )
            ],
        )
        seqs.append(
            GameSequence(
                episode_id=f"smoke-seq-{i}",
                seat=0,
                archetype="core",
                opp_archetype="iono",
                deck=[1] * 60,
                value=1.0 if i % 2 == 0 else -1.0,
                decisions=[dec],
                policy_targets=None,
                factorized_policy_targets=None,
                target_provenance={"pure_rl": True, "soft_policy_targets": False},
            )
        )
    return BootstrapDataset(sequences=seqs)


def _ensure_pure_rl_checkpoint(path: Path, seed: int, *, smoke: bool = False) -> Path:
    """Build or validate a fresh small Pure-RL seed (not AZ / not starter prior)."""
    import torch
    from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
    from poke_bot.train import load_model_from_checkpoint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Abhyuday: "The starter is terrible." — never CE-clone competition starter.
    banned = ("starter", "sample_submission", "rl_starter", "kaggle_starter")
    name_l = path.name.lower()
    path_l = str(path).lower()
    if any(tok in name_l or tok in path_l for tok in banned):
        raise SystemExit(
            f"PURE_RL refuse competition starter prior at {path}; "
            "use a fresh small pure_rl seed (Abhyuday: starter is terrible)"
        )
    if path.is_file():
        _checkpoint_contract(path, smoke=smoke)
        model = load_model_from_checkpoint(path, device=torch.device("cpu"))
        n = count_params(model)
        print(f"[pure_rl] loaded checkpoint params={n} path={path}", flush=True)
        validate_param_budget(n, fail_max=int(config.PURE_RL.param_fail_max))
        cfg = getattr(model, "cfg", None)
        # Pure-RL overnight is d_model=96 L4/4/4 (~1.77M). Hope/legacy giants
        # are d_model>=192 — refuse those, not the intentional dense seed.
        if cfg is not None and int(getattr(cfg, "d_model", 0)) >= 192:
            raise SystemExit(
                f"PURE_RL refuse Hope-sized checkpoint d_model={cfg.d_model} "
                f"at {path}; pass a small pure_rl seed or omit --base-checkpoint"
            )
        return path

    torch.manual_seed(seed)
    cfg = pure_rl_model_config(**({"dropout": 0.0} if smoke else {}))
    if smoke:
        # Tiny vocabs for CPU canary speed; still ≤3.5M with real vocab too.
        model = build_pure_rl_model(
            device=torch.device("cpu"),
            cfg=cfg,
            validate=True,
            aux_archetype_classes=8,
            encoder_vocab=128,
            decoder_vocab=128,
            belief_card_vocab=128,
        )
    else:
        model = build_pure_rl_model(device=torch.device("cpu"), cfg=cfg, validate=True)
    n = count_params(model)
    atomic_torch_save(
        build_checkpoint(
            model=model,
            step=0,
            epoch=0,
            model_config=cfg,
            extra={
                "pure_rl": True,
                "smoke": smoke,
                "param_count": n,
                "model_profile": model_config_dict(cfg),
            },
        ),
        path,
    )
    _checkpoint_contract(path, smoke=smoke)
    print(f"[pure_rl] wrote small seed params={n} path={path}", flush=True)
    return path


def _collect_temperature(args: argparse.Namespace, iteration: int) -> float:
    t0 = float(args.collect_temperature)
    t1 = float(getattr(config.PURE_RL, "collect_temperature_final", 0.7))
    anneal = max(1, int(getattr(config.PURE_RL, "temperature_anneal_iters", 50)))
    if iteration >= anneal:
        return t1
    frac = float(iteration) / float(anneal)
    return t0 + (t1 - t0) * frac


def _write_metrics(run_dir: Path, it: int, metrics: IterationMetrics) -> None:
    out = run_dir / "metrics" / f"iter_{it:05d}.json"
    _write_json_exclusive(out, metrics_to_dict(metrics))
    latest = run_dir / "metrics" / "latest.json"
    _atomic_json(latest, metrics_to_dict(metrics))


def _tqdm_mininterval() -> float:
    """Throttle bar redraws; floor 1s so file viewers stay calm."""
    raw = os.environ.get("PURE_RL_TQDM_MININTERVAL", "1.5")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 1.5


def _heartbeat_sec() -> float:
    """Sparse stdout heartbeat so the monitor log does not look stalled."""
    raw = os.environ.get("PURE_RL_HEARTBEAT_SEC", "300")
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 300.0


def _progress_log_path() -> Path:
    return Path(
        os.environ.get(
            "PURE_RL_PROGRESS_LOG",
            str((ROOT / "outputs/logs/pure_rl_core.progress.log").resolve()),
        )
    )


def _progress_status_path(progress_log: Optional[Path] = None) -> Path:
    """Sibling single-line status file rewritten each bar tick."""
    p = progress_log or _progress_log_path()
    name = p.name
    if name.endswith(".progress.log"):
        return p.with_name(name[: -len(".progress.log")] + ".progress.status")
    return p.with_name(p.stem + ".progress.status")


def _progress_snapshot_sec() -> float:
    """How often to emit an archival newline into progress.log (0=off)."""
    raw = os.environ.get("PURE_RL_PROGRESS_SNAPSHOT_SEC", "300")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


def _tqdm_inplace_enabled() -> bool:
    raw = os.environ.get("PURE_RL_TQDM_INPLACE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _public_mix_local_only() -> bool:
    """Public mix (baseline vs public/roster opponents) finishes fast locally.

    Default ON: keep the light public-mix slice entirely on local workers so
    additive remotes (Elmo/bert) stay free for the (much larger) self-play
    wave and any other/slower work. Set 0/false to fall back to the shared
    heavy-local frac split (``PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC``) instead of
    excluding remotes outright. Self-play routing is untouched either way.
    """
    raw = os.environ.get("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _public_mix_min_local_frac() -> float:
    """Heavy-local fallback frac when ``PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0``."""
    raw = os.environ.get("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", "0.95")
    try:
        frac = float(raw)
    except ValueError:
        frac = 0.95
    return min(0.95, max(0.05, frac))


class _InPlaceProgressStream:
    """TTY-like stderr so tqdm keeps \\r in-place (no \\r→\\n spam).

    ``tail -F`` on an append-only file cannot show a true single line; use
    ``scripts/watch_pure_rl_progress.sh``, ``watch -n1 cat *.progress.status``,
    or ``less -r +F *.progress.log``. Sparse newlines are emitted on bar close
    (tqdm) and every ``PURE_RL_PROGRESS_SNAPSHOT_SEC`` for archival crumbs.
    """

    __slots__ = (
        "_stream",
        "_status_path",
        "_last_visible",
        "_last_snapshot",
        "_snapshot_sec",
    )

    def __init__(self, stream: Any, *, status_path: Path) -> None:
        self._stream = stream
        self._status_path = status_path
        self._last_visible = ""
        self._last_snapshot = time.time()
        self._snapshot_sec = _progress_snapshot_sec()

    def write(self, s: str) -> int:
        if not s:
            return 0
        written = self._stream.write(s)
        flush = getattr(self._stream, "flush", None)
        if flush is not None:
            flush()
        chunk = s.replace("\n", "\r")
        if "\r" in chunk:
            part = chunk.rsplit("\r", 1)[-1]
        else:
            part = chunk
        visible = part.strip()
        if visible:
            self._last_visible = visible
            try:
                self._status_path.parent.mkdir(parents=True, exist_ok=True)
                self._status_path.write_text(visible + "\n", encoding="utf-8")
            except OSError:
                pass
        if (
            self._snapshot_sec > 0
            and "\n" not in s
            and self._last_visible
            and (time.time() - self._last_snapshot) >= self._snapshot_sec
        ):
            self._last_snapshot = time.time()
            self._stream.write("\n")
            if flush is not None:
                flush()
        return written

    def flush(self) -> None:
        flush = getattr(self._stream, "flush", None)
        if flush is not None:
            flush()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return int(self._stream.fileno())


class _TqdmProgress:
    """Collect/heldout bar on stderr; rare stdout heartbeat for monitors."""

    def __init__(
        self,
        *,
        stage: str,
        iteration: int,
        total: int,
        remotes: int = 0,
        inplace: Optional[bool] = None,
        mininterval: Optional[float] = None,
    ) -> None:
        from tqdm.auto import tqdm

        self.stage = stage
        self.iteration = iteration
        self.total = max(0, int(total))
        self.remotes = int(remotes)  # live open remote sockets (dispatch)
        self.remote_demand: Optional[int] = None  # scheduler demand (may lag sockets)
        self.wr: Optional[str] = None  # live win-rate readout (heldout gate)
        self._t0 = time.time()
        self._last_heartbeat = self._t0
        self._done = 0
        self._decisions = 0
        err = sys.stderr
        to_file = not getattr(err, "isatty", lambda: False)()
        progress_hint = _progress_log_path()
        status_hint = _progress_status_path(progress_hint)
        use_inplace = _tqdm_inplace_enabled() if inplace is None else bool(inplace)
        bar_interval = (
            float(mininterval) if mininterval is not None else _tqdm_mininterval()
        )
        bar_file: Any = err
        if to_file and use_inplace:
            bar_file = _InPlaceProgressStream(err, status_path=status_hint)
        # Force a visible fill bar ({bar:N}). Default ascii+long postfix crushed
        # the meter to "|2|" in *.progress.status / watch.
        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar:36}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        self._bar = tqdm(
            total=self.total,
            desc=f"pure_rl {stage} iter={iteration}",
            unit="game",
            file=bar_file,
            mininterval=max(1.0, bar_interval),
            miniters=1,
            ascii=False,  # unicode █/░ — readable in iTerm + status mirror
            bar_format=bar_format,
            dynamic_ncols=False,
            # The formal heldout postfix includes a two-decimal WR, game count,
            # and gate target.  140 columns clipped the second decimal in the
            # canonical status/tail file, so reserve enough fixed width for the
            # complete readout while retaining the 36-cell progress bar.
            ncols=168 if to_file else 140,
            leave=True,
            smoothing=0.1,
        )
        self._bar.set_postfix(**self._postfix(sps="0"))
        if to_file:
            mode = "in-place" if use_inplace else "line-spam"
            print(
                f"[pure_rl] tqdm games bar {mode} on stderr "
                f"(stage={stage} iter={iteration} total={self.total}); "
                f"remotes=active_sockets (rdmd=demand when desynced); "
                f"watch: bash scripts/watch_pure_rl_progress.sh "
                f"(or: watch -n1 cat {status_hint}; less -r +F {progress_hint})",
                flush=True,
            )

    def _postfix(self, *, sps: str) -> dict[str, Any]:
        """``remotes`` = live open sockets; ``rdmd`` only when ≠ demand."""
        out: dict[str, Any] = {"remotes": int(self.remotes), "sps": sps}
        if self.wr is not None:
            out["wr"] = self.wr
        dem = self.remote_demand
        if dem is not None and int(dem) != int(self.remotes):
            out["rdmd"] = int(dem)
        return out

    def set_remotes(
        self, active: int, *, demand: Optional[int] = None
    ) -> None:
        """Hot-update bar when mid-iter demand grows/shrinks dispatch sockets."""
        self.remotes = max(0, int(active))
        if demand is not None:
            self.remote_demand = max(0, int(demand))
        try:
            # Keep last sps token if present; tick() refreshes on next game.
            prev = getattr(self._bar, "postfix", None) or {}
            sps = str(prev.get("sps", "0")) if isinstance(prev, dict) else "0"
            self._bar.set_postfix(**self._postfix(sps=sps), refresh=True)
        except Exception:
            pass

    def set_wr(
        self, win_rate: float, games: int, *, target: Optional[float] = None
    ) -> None:
        """Live win-rate readout — heldout gate games report outcomes eagerly.

        Called as each heldout game result lands, so the bar shows WR
        climbing/settling in real time instead of only at iter-end.
        """
        if games <= 0:
            return
        label = f"{win_rate:.2%}/{int(games)}g"
        if target is not None:
            label += f" (gate {target:.0%})"
        self.wr = label
        try:
            prev = getattr(self._bar, "postfix", None) or {}
            sps = str(prev.get("sps", "0")) if isinstance(prev, dict) else "0"
            self._bar.set_postfix(**self._postfix(sps=sps), refresh=True)
        except Exception:
            pass

    def tick(self, *, decisions: int = 0) -> None:
        self._done += 1
        self._decisions = int(decisions)
        elapsed = max(time.time() - self._t0, 1e-6)
        sps = self._decisions / elapsed
        # Postfix before update so the written status line includes sps.
        self._bar.set_postfix(
            **self._postfix(sps=f"{sps:.1f}"), refresh=False
        )
        self._bar.update(1)
        now = time.time()
        if (now - self._last_heartbeat) >= _heartbeat_sec():
            self._last_heartbeat = now
            gps = self._done / elapsed
            dem = (
                f" rdmd={self.remote_demand}"
                if self.remote_demand is not None
                and int(self.remote_demand) != int(self.remotes)
                else ""
            )
            print(
                f"[pure_rl] heartbeat stage={self.stage} iter={self.iteration} "
                f"games={self._done}/{self.total} "
                f"gps={gps:.2f} sps={sps:.1f} remotes={self.remotes}{dem}",
                flush=True,
            )

    def close(self) -> None:
        try:
            self._bar.close()
        except Exception:
            pass


def _record_to_compact_game(record: dict[str, Any]) -> Optional[CompactGame]:
    """Strip soft behavior π; keep selected_index + observation for AWR."""
    steps = list(record.get("steps") or [])
    if not steps:
        return None
    soft = list(record.get("factorized_policy_targets") or [])
    decisions: list[CompactDecision] = []
    for i, step in enumerate(steps):
        sel = 0
        n_opt = 1
        if i < len(soft) and soft[i]:
            row0 = soft[i][0] if isinstance(soft[i], list) else soft[i]
            if isinstance(row0, dict):
                sel = int(row0.get("selected_index", 0))
                combos = row0.get("action_combos") or []
                n_opt = max(len(combos), sel + 1, 1)
        decisions.append(
            CompactDecision(
                env_step=int(step.get("env_step", i)),
                selected_index=sel,
                n_options=n_opt,
                action=[int(x) for x in (step.get("action") or [])],
                observation=dict(step.get("observation") or {}),
                aux_labels=dict(step.get("aux_labels") or {}),
            )
        )
    if not decisions:
        return None
    return CompactGame(
        episode_id=str(record.get("episode_id") or f"pure-rl-{time.time_ns()}"),
        seat=int(record.get("seat") or 0),
        archetype=str(record.get("archetype") or "core"),
        opp_archetype=str(record.get("opp_archetype") or ""),
        deck=[int(x) for x in (record.get("deck") or [])],
        value=float(record.get("value") or 0.0),
        decisions=decisions,
        source="pure_rl",
        target_provenance={
            **dict(record.get("target_provenance") or {}),
            "pure_rl": True,
            "soft_policy_targets": False,
        },
    )


def _resolve_remote_endpoints(args: argparse.Namespace) -> list[str]:
    from poke_bot.remote_jobs import expand_endpoint_specs

    if args.smoke or args.no_remote_workers:
        return []
    raw = args.remote_worker_endpoints
    if raw is None:
        raw = os.environ.get("PURE_RL_REMOTE_WORKER_ENDPOINTS")
        if raw is None:
            raw = os.environ.get(
                "POKEBOT_REMOTE_WORKER_ENDPOINTS", DEFAULT_REMOTE_ENDPOINTS
            )
    if not str(raw).strip():
        return []
    return expand_endpoint_specs([str(raw)])


def _our_decks(
    mode: str, specialist_archetype: Optional[str] = None
) -> list[tuple[str, list[int]]]:
    """Stage A core: wide multi-archetype pool (baselines + archetype-samples).

    A specialist must name one exact deck from the immutable ladder
    representative catalog. Core must not collapse to ``default_pool()``
    (dragapult + hammer-pult only).
    """
    from poke_bot.deck_pool import default_pool, primary_deck, primary_archetype, read_deck
    from poke_bot import paths as _paths

    if mode == "specialist":
        requested = str(specialist_archetype or "").strip().lower()
        if not requested:
            raise ValueError("specialist mode requires an explicit archetype")
        pinned, _mix, _representatives, _contract = _core_ladder_decks()
        matches = [
            (name, list(cards))
            for name, cards in pinned
            if str(name).strip().lower() == requested
        ]
        if len(matches) != 1:
            raise ValueError(
                f"specialist archetype {requested!r} is not one exact pinned "
                f"ladder representative; available={sorted(name for name, _ in pinned)}"
            )
        return matches

    out: list[tuple[str, list[int]]] = []
    seen_names: set[str] = set()
    seen_lists: set[tuple[int, ...]] = set()

    def _add(name: str, deck: list[int]) -> None:
        n = str(name).strip()
        if not n or n in seen_names:
            return
        key = tuple(int(x) for x in deck)
        if len(key) != 60 or key in seen_lists:
            return
        seen_names.add(n)
        seen_lists.add(key)
        out.append((n, list(deck)))

    # 1) All installable baseline decks from baselines/manifest.json
    try:
        from poke_bot.baselines_runtime import ensure_baselines_installed, load_manifest

        for spec in ensure_baselines_installed(load_manifest()):
            try:
                if spec.deck_csv.is_file():
                    _add(spec.id, read_deck(spec.deck_csv))
            except Exception:
                continue
    except Exception:
        pass

    # 2) decks/archetype-samples/ (dozens of archetype variants)
    samples_dir = _paths.DECKS_DIR / "archetype-samples"
    if samples_dir.is_dir():
        for path in sorted(samples_dir.glob("*.csv")):
            try:
                _add(path.stem, read_deck(path))
            except Exception:
                continue

    # 3) Keep default_pool entries when they add a unique list/name
    try:
        pool = default_pool()
        for name in pool.names():
            try:
                _add(name, pool.get(name).load())
            except Exception:
                continue
    except Exception:
        pass

    if not out:
        out.append((primary_archetype(), primary_deck()))
    return out


def _select_measurement_decks(
    decks: list[tuple[str, list[int]]], requested: str | None
) -> list[tuple[str, list[int]]]:
    """Resolve a strict, ordered evaluation-only subset of the training pool."""
    tokens = [token.strip().lower() for token in str(requested or "").split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        return [(name, list(cards)) for name, cards in decks]
    if len(tokens) != len(set(tokens)):
        raise ValueError("--measurement-decks contains duplicate deck IDs")
    by_id = {str(name).strip().lower(): (name, cards) for name, cards in decks}
    missing = [token for token in tokens if token not in by_id]
    if missing:
        raise ValueError(
            "--measurement-decks contains IDs outside the active training pool: "
            f"{missing}; available={sorted(by_id)}"
        )
    return [(by_id[token][0], list(by_id[token][1])) for token in tokens]


def _core_ladder_decks() -> tuple[
    list[tuple[str, list[int]]], Any, Any, dict[str, Any]
]:
    """Load the pinned top-ladder family mix and exact modal representatives."""
    from poke_bot.ladder_deck_mix import (
        load_ladder_deck_mix,
        load_ladder_deck_representatives,
    )

    mix = load_ladder_deck_mix(LADDER_DECK_MIX_PATH)
    representatives = load_ladder_deck_representatives(
        LADDER_DECK_REPRESENTATIVES_PATH
    )
    bound = representatives.bind(mix)
    decks = [
        (entry.bucket.deck_id, list(entry.card_ids))
        for entry in bound
    ]
    contract = representatives.contract(mix)
    return decks, mix, representatives, contract


def _derange_deck_schedule(
    ours: tuple[str, ...], opponents: tuple[str, ...]
) -> tuple[str, ...]:
    """Preserve exact quotas and minimize same-family self-play deterministically.

    Both schedules are sampled from the same ladder quotas.  Grouping equal
    families and rotating the opponent tokens by the largest quota constructs
    a multiset derangement whenever one is possible (``2 * max_quota <= n``).
    When a tiny wave makes that impossible, the same construction leaves the
    theoretical minimum ``2 * max_quota - n`` same-family pairs instead of
    crashing the unattended trainer.
    """
    from collections import Counter

    if len(ours) != len(opponents):
        raise ValueError("self-play deck schedules have different lengths")
    n = len(ours)
    if n <= 0:
        return ()

    our_counts = Counter(ours)
    opponent_counts = Counter(opponents)
    if our_counts != opponent_counts:
        raise ValueError(
            "self-play deck schedules must have identical ladder quotas"
        )

    # Keeping the largest group first makes a shift by its size disjoint from
    # every group when a full derangement is feasible.  Label is the stable
    # tie-break, so the result is reproducible across Python processes.
    families = sorted(our_counts, key=lambda family: (-our_counts[family], family))
    grouped_positions = [
        index
        for family in families
        for index, scheduled_family in enumerate(ours)
        if scheduled_family == family
    ]
    grouped_opponents = [
        family for family in families for _ in range(opponent_counts[family])
    ]
    shift = max(our_counts.values())
    rotated = grouped_opponents[shift:] + grouped_opponents[:shift]
    candidate = [""] * n
    for position, opponent in zip(grouped_positions, rotated):
        candidate[position] = opponent
    return tuple(candidate)


def _spec_payload(spec) -> dict[str, Any]:
    from poke_bot.baselines_runtime import baseline_spec_payload

    return baseline_spec_payload(spec)


class _LeafFarm:
    """Local dual-GPU leaf servers for host CPU workers."""

    def __init__(self) -> None:
        self.procs: list = []
        self.req_qs: list = []
        self.ctrl_qs: list = []
        self.status_qs: list = []
        self.alive_evts: list = []
        self.resp_qs: list = []
        self.remote_channel = None
        self.version = 0
        self.digest: Optional[str] = None

    def start(
        self,
        *,
        ckpt: Path,
        digest: str,
        leaf_devices: list[int],
        n_workers: int,
        max_batch: Optional[int],
        coalesce_ms: float,
    ) -> None:
        import multiprocessing as mp
        from queue import Empty

        from poke_bot.batched_infer import run_leaf_server

        # Rebuild-safe: never append onto stale queues from a prior farm.
        if self.procs or self.status_qs:
            self.stop()

        self.digest = digest
        self.version = 0
        mpctx = mp.get_context("spawn")
        # Leaf replicas may exceed CPU worker slots (VRAM-packing for util).
        n_servers = max(1, len(leaf_devices))
        devices = leaf_devices[:n_servers]
        self.procs = []
        self.req_qs = []
        self.ctrl_qs = []
        self.status_qs = []
        self.alive_evts = []
        self.resp_qs = [mpctx.Queue(maxsize=2) for _ in range(n_workers)]
        slot_counter = mpctx.Value("i", 0)
        readies = []
        for j, dev in enumerate(devices):
            rq = mpctx.Queue(maxsize=64)
            cq = mpctx.Queue(maxsize=8)
            sq = mpctx.Queue(maxsize=16)
            ev = mpctx.Event()
            alive = mpctx.Event()
            proc = mpctx.Process(
                target=run_leaf_server,
                args=(str(ckpt), f"cuda:{dev}", rq, self.resp_qs),
                kwargs=dict(
                    ready_evt=ev,
                    alive_evt=alive,
                    ctrl_q=cq,
                    status_q=sq,
                    expected_digest=digest,
                    initial_version=self.version,
                    bf16=True,
                    max_batch=max_batch,
                    coalesce_ms=coalesce_ms,
                ),
                daemon=True,
            )
            proc.start()
            self.procs.append(proc)
            self.req_qs.append(rq)
            self.ctrl_qs.append(cq)
            self.status_qs.append(sq)
            self.alive_evts.append(alive)
            readies.append(ev)
        for j, ev in enumerate(readies):
            if not ev.wait(timeout=240):
                self.stop()
                raise RuntimeError(f"leaf server {j} not ready in 240s")
            try:
                status = self.status_qs[j].get(timeout=60)
            except Empty as exc:
                self.stop()
                raise RuntimeError(
                    f"leaf server {j} ready without status ack (device={devices[j]})"
                ) from exc
            if not status.get("ok") or not self.procs[j].is_alive():
                self.stop()
                raise RuntimeError(f"leaf server {j} bad ready ack: {status}")
        self.remote_channel = {
            "req_qs": self.req_qs,
            "resp_qs": self.resp_qs,
            "slot_counter": slot_counter,
            "ctrl_qs": self.ctrl_qs,
            "generation": 0,
            "alive_evts": self.alive_evts,
            "expected_digest": digest,
            "expected_version": self.version,
            "timeout_s": config.SEARCH.remote_request_timeout_s,
            # Even-spread device map + GPU0-biased sticky / least-queue feed.
            "leaf_devices": list(devices),
            "gpu0_client_frac": float(
                os.environ.get("PURE_RL_GPU0_CLIENT_FRAC", "0.38")
            ),
        }
        print(
            f"[pure_rl] leaf-eval=gpu-server x{len(self.procs)} devices={devices} "
            f"workers={n_workers} coalesce_ms={coalesce_ms} "
            f"gpu0_client_frac={self.remote_channel['gpu0_client_frac']:.2f}",
            flush=True,
        )

    def reload(self, ckpt: Path, digest: str) -> None:
        """Reload all leaf servers; hard-fail unless every ack matches ``digest``."""
        if not self.ctrl_qs:
            return
        requested = self.version + 1
        for cq in self.ctrl_qs:
            cq.put(
                {
                    "cmd": "reload",
                    "path": str(ckpt),
                    "digest": digest,
                    "version": requested,
                }
            )
        mismatches: list[str] = []
        for i, sq in enumerate(self.status_qs):
            try:
                status = sq.get(timeout=240)
            except Exception as exc:
                mismatches.append(f"leaf[{i}] status timeout/error: {exc}")
                continue
            if status.get("type") != "reload" or not status.get("ok"):
                mismatches.append(f"leaf[{i}] bad ack: {status!r}")
                continue
            got = status.get("checkpoint_digest")
            if got != digest:
                mismatches.append(
                    f"leaf[{i}] digest mismatch: expected {digest}, got {got}"
                )
        if mismatches:
            raise RuntimeError(
                "leaf reload hard-gate failed: " + "; ".join(mismatches)
            )
        self.version = requested
        self.digest = digest
        if self.remote_channel is not None:
            self.remote_channel["expected_digest"] = digest
            self.remote_channel["expected_version"] = self.version

    def stop(self) -> None:
        procs = list(self.procs)
        queues = [
            *self.req_qs,
            *self.ctrl_qs,
            *self.status_qs,
            *self.resp_qs,
        ]
        for cq in self.ctrl_qs:
            try:
                cq.put({"cmd": "stop"})
            except Exception:
                pass
        for proc in procs:
            try:
                proc.join(timeout=5)
            except Exception:
                pass
            try:
                alive = bool(proc.is_alive())
            except Exception:
                alive = False
            if alive:
                try:
                    proc.terminate()
                except Exception:
                    pass
                # Reap a force-terminated child.  Omitting this second join
                # leaked process handles and named multiprocessing semaphores
                # on every leaf-farm rebuild.
                try:
                    proc.join(timeout=5)
                except Exception:
                    pass
            try:
                alive = bool(proc.is_alive())
            except Exception:
                alive = False
            if alive and hasattr(proc, "kill"):
                try:
                    proc.kill()
                    proc.join(timeout=5)
                except Exception:
                    pass
            try:
                if not proc.is_alive() and hasattr(proc, "close"):
                    proc.close()
            except Exception:
                pass
        for q in queues:
            close_mp_queue(q)
        self.procs.clear()
        self.req_qs.clear()
        self.ctrl_qs.clear()
        self.status_qs.clear()
        self.alive_evts.clear()
        self.resp_qs.clear()
        self.remote_channel = None


def run_smoke_loop(args: argparse.Namespace) -> int:
    hw = full_hardware_profile()
    hw = replace(
        hw,
        allow_single_gpu=True,
        leaf_gpu0_replicas=max(1, hw.leaf_gpu0_replicas),
        leaf_gpu1_replicas=max(1, hw.leaf_gpu1_replicas),
    )
    hw.validate_or_raise(visible_gpu_count=1)

    run_dir = _run_dir(args.run_name, smoke=True)
    ckpt = args.base_checkpoint or (run_dir / "checkpoints" / "seed.pt")
    ckpt = _ensure_pure_rl_checkpoint(Path(ckpt), args.seed, smoke=True)

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    adv_hist: list[float] = []
    agr_hist: list[float] = []

    manifest = {
        "run_name": args.run_name,
        "mode": args.mode,
        "smoke": True,
        "hardware": hw.as_dict(),
        "stage": stage_to_dict(stage),
        "created": time.time(),
        "gate_wr": float(args.gate_wr),
        "heldout_games": int(args.heldout_games),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for it in range(args.iterations):
        t0 = time.time()
        next_shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
        writer = CompactShardWriter(next_shard)
        n_games = args.smoke_games
        collect_future_games = _smoke_games(
            n_games,
            seed=args.seed + it,
            archetype=(
                "core" if args.mode == "core" else str(args.specialist_archetype)
            ),
        )

        def _collect() -> None:
            writer.write_games(collect_future_games)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_collect)
            dataset = _smoke_dataset(n_games, args.seed + it)
            train_cfg = TrainConfig.pure_rl_defaults(
                epochs=max(1, args.train_epochs),
                seed=args.seed + it,
                aux_loss_weight=float(args.archetype_aux_loss_weight),
                opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
                lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
                prize_race_loss_weight=float(args.prize_race_loss_weight),
                alakazam_guide_loss_weight=float(
                    args.alakazam_guide_loss_weight
                ),
            )
            import torch
            from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
            from poke_bot.train import load_model_from_checkpoint, batch_losses

            model = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
            total, bm = batch_losses(
                model,
                list(dataset.sequences),
                value_weight=1.0,
                aux_weight=train_cfg.aux_loss_weight,
                opp_hand_weight=train_cfg.opp_hand_loss_weight,
                opp_remainder_weight=train_cfg.opp_remainder_loss_weight,
                lethal_threat_weight=train_cfg.lethal_threat_loss_weight,
                prize_race_weight=train_cfg.prize_race_loss_weight,
                alakazam_guide_weight=train_cfg.alakazam_guide_loss_weight,
                pure_rl=True,
                awr_beta=train_cfg.awr_beta,
                awr_weight_max=train_cfg.awr_weight_max,
            )
            if bm.n_decisions > 0 and torch.isfinite(total):
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            atomic_torch_save(
                build_checkpoint(
                    model=model,
                    step=it + 1,
                    epoch=1,
                    model_config=getattr(model, "cfg", None),
                    extra={"pure_rl": True, "iteration": it, "mode": args.mode},
                ),
                out_ckpt,
            )
            ckpt = out_ckpt
            fut.result()

        thr = writer.throughput()
        smoke_heldout_games = int(args.heldout_games or 200)
        rows = [
            {
                "opponent_id": oid,
                "our_seat": 0,
                "winner": 0,
                "baseline_failed": False,
            }
            for oid in OFFICIAL_BASELINE_IDS
            for _ in range(
                max(1, smoke_heldout_games // len(OFFICIAL_BASELINE_IDS))
            )
        ]
        gate = aggregate_heldout_wr(
            rows, target_wr=args.gate_wr, min_games=smoke_heldout_games
        )
        adv_hist.append(float(bm.mean_advantage))
        agr_hist.append(0.5)
        abort = evaluate_aborts(
            mean_advantages=adv_hist, policy_prev_agreements=agr_hist, k=3
        )
        elapsed = max(time.time() - t0, 1e-6)
        metrics = IterationMetrics(
            iteration=it,
            stage=stage.stage.value,
            games=writer.n_games,
            decisions=writer.n_decisions,
            games_per_sec=thr["games_per_sec"],
            decisions_per_sec=thr["decisions_per_sec"],
            games_per_hour=thr["games_per_sec"] * 3600.0,
            mean_return=float(bm.target_value_mean),
            mean_advantage=float(bm.mean_advantage),
            awr_weight_mean=float(bm.awr_weight_mean),
            awr_weight_p50=float(bm.awr_weight_p50),
            awr_weight_p95=float(bm.awr_weight_p95),
            awr_weight_clip_frac=float(bm.awr_weight_clip_frac),
            policy_selected_nll=float(bm.policy_selected_nll),
            policy_prev_agreement=0.5,
            self_distill_flag=abort.self_distill_flag,
            heldout_wr=gate.win_rate,
            heldout_games=gate.games,
            gate_passed=gate.passed and not abort.abort,
            extra={
                "abort": asdict(abort),
                "elapsed_sec": elapsed,
                "hardware": hw.as_dict(),
                "checkpoint": str(ckpt),
            },
        )
        _write_metrics(run_dir, it, metrics)
        print(
            f"[pure_rl smoke] iter={it} games={metrics.games} "
            f"awr_w={metrics.awr_weight_mean:.3f} heldout_wr={gate.win_rate:.3f} "
            f"gate={gate.passed} abort={abort.abort}",
            flush=True,
        )
        if gate.passed and not abort.abort:
            if args.mode == "core":
                (run_dir / "CORE_GATE_PASSED").write_text(
                    json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
                )
                print("[pure_rl] CORE GATE PASSED", flush=True)
                break
            (run_dir / "SPECIALIST_GATE_PASSED").write_text(
                json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
            )
            print("[pure_rl] SPECIALIST GATE PASSED", flush=True)
            break
        if abort.abort:
            print(f"[pure_rl] abort promote: {abort.reason}", flush=True)
            return 2
    return 0


def _latest_official_heldout_win_rates(
    state: dict[str, Any],
    opponent_ids: tuple[str, ...] = OFFICIAL_BASELINE_IDS,
) -> dict[str, float]:
    """Return the newest exact heldout rates that cover every official ID."""
    wanted = tuple(str(opponent_id) for opponent_id in opponent_ids)
    candidates: list[Any] = []
    history = state.get("history") if isinstance(state, dict) else None
    if isinstance(history, list):
        for entry in reversed(history):
            if not isinstance(entry, dict) or not bool(entry.get("completed")):
                continue
            candidates.extend(
                [entry.get("raw_heldout_gate"), entry.get("heldout_gate")]
            )
    candidates.append(
        state.get("heldout_champion_evidence") if isinstance(state, dict) else None
    )
    for candidate in candidates:
        per_opponent = (
            candidate.get("per_opponent")
            if isinstance(candidate, dict)
            else None
        )
        if not isinstance(per_opponent, dict):
            continue
        rates: dict[str, float] = {}
        for opponent_id in wanted:
            row = per_opponent.get(opponent_id)
            if not isinstance(row, dict) or float(row.get("games") or 0) <= 0:
                break
            value = float(row.get("win_rate", row.get("wr", -1.0)))
            if not 0.0 <= value <= 1.0:
                break
            rates[opponent_id] = value
        if len(rates) == len(wanted):
            return rates
    return {}


def _adaptive_official_target_weights(
    opponent_ids: tuple[str, ...],
    win_rates: dict[str, float],
    *,
    target_win_rate: float,
    minimum_share: float,
    gap_power: float,
) -> dict[str, float]:
    """Blend a non-starvation floor with powered exact-heldout deficits."""
    ids = tuple(str(opponent_id) for opponent_id in opponent_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("official target IDs must be non-empty and unique")
    floor = float(minimum_share)
    power = float(gap_power)
    target = float(target_win_rate)
    if not 0.0 <= floor <= 1.0 / len(ids):
        raise ValueError("minimum official target share is infeasible")
    if not 0.0 < power <= 4.0 or not 0.0 <= target <= 1.0:
        raise ValueError("invalid official target gap configuration")
    if set(win_rates) != set(ids) or any(
        not 0.0 <= float(win_rates[opponent_id]) <= 1.0
        for opponent_id in ids
    ):
        return {opponent_id: 1.0 / len(ids) for opponent_id in ids}
    deficits = {
        opponent_id: max(0.0, target - float(win_rates[opponent_id])) ** power
        for opponent_id in ids
    }
    deficit_total = sum(deficits.values())
    if deficit_total <= 1e-12:
        return {opponent_id: 1.0 / len(ids) for opponent_id in ids}
    adaptive_mass = 1.0 - floor * len(ids)
    weights = {
        opponent_id: floor
        + adaptive_mass * deficits[opponent_id] / deficit_total
        for opponent_id in ids
    }
    # Normalize once to remove floating-point accumulation without changing
    # the relative allocation used by the integer quota scheduler.
    total = sum(weights.values())
    return {opponent_id: weights[opponent_id] / total for opponent_id in ids}


def _interleaved_opponent_schedule(
    n_games: int,
    *,
    priority_specs: list[Any],
    diverse_specs: list[Any],
    priority_frac: float,
    seed: int,
    iteration: int,
    priority_weights: Optional[dict[str, float]] = None,
) -> tuple[tuple[Any, str], ...]:
    """Build an exact, evenly interleaved target/diverse opponent schedule.

    Training against a known public baseline is not a formal-evaluation row:
    the caller supplies a disjoint seed range for the later greedy gate.  This
    schedule merely controls which policy produces experience.  Exact group
    quotas and deterministic rotations make the immutable run contract easy
    to audit while avoiding a large all-target then all-diverse burst.
    """
    total = max(0, int(n_games))
    if total <= 0:
        return ()
    priority = list(priority_specs)
    diverse = list(diverse_specs)
    if not priority and not diverse:
        return ()
    if not priority:
        n_priority = 0
    elif not diverse:
        n_priority = total
    else:
        n_priority = int(round(total * min(1.0, max(0.0, float(priority_frac)))))
    n_diverse = total - n_priority

    def _rotation(rows: list[Any], label: str) -> int:
        if not rows:
            return 0
        token = f"{int(seed)}:{int(iteration)}:{label}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % len(rows)

    p_offset = _rotation(priority, "official_target")
    d_offset = _rotation(diverse, "diverse_public")
    priority_sequence: list[Any] = []
    if priority_weights is not None and n_priority > 0:
        ids = [str(spec.id) for spec in priority]
        if len(set(ids)) != len(ids):
            raise ValueError("weighted official target specs must be unique")
        if set(priority_weights) != set(ids):
            raise ValueError("official target weights must cover the exact roster")
        raw = [float(priority_weights[opponent_id]) for opponent_id in ids]
        if any(not math.isfinite(value) or value < 0.0 for value in raw):
            raise ValueError("official target weights must be finite and nonnegative")
        raw_total = sum(raw)
        if raw_total <= 0.0:
            raise ValueError("official target weights must contain positive mass")
        ideals = [n_priority * value / raw_total for value in raw]
        quotas = [int(math.floor(value)) for value in ideals]
        remaining = n_priority - sum(quotas)
        tie_rank = {
            (p_offset + index) % len(priority): index
            for index in range(len(priority))
        }
        remainder_order = sorted(
            range(len(priority)),
            key=lambda index: (
                -(ideals[index] - quotas[index]),
                tie_rank[index],
            ),
        )
        for index in remainder_order[:remaining]:
            quotas[index] += 1
        # Smooth weighted round robin spreads the exact Hamilton quotas across
        # the wave instead of emitting a large weakest-opponent burst.
        current = [0] * len(priority)
        order = [
            (p_offset + index) % len(priority)
            for index in range(len(priority))
        ]
        order_rank = {index: rank for rank, index in enumerate(order)}
        for _ in range(n_priority):
            for index, quota in enumerate(quotas):
                current[index] += quota
            selected = max(
                range(len(priority)),
                key=lambda index: (current[index], -order_rank[index]),
            )
            current[selected] -= n_priority
            priority_sequence.append(priority[selected])
        assert {
            str(spec.id): sum(str(row.id) == str(spec.id) for row in priority_sequence)
            for spec in priority
        } == {str(spec.id): quotas[index] for index, spec in enumerate(priority)}
    p_index = d_index = 0
    schedule: list[tuple[Any, str]] = []
    for position in range(total):
        # Cumulative integer quotas spread target rows through the whole wave.
        take_priority = (
            ((position + 1) * n_priority) // total
            > (position * n_priority) // total
        )
        if take_priority:
            spec = (
                priority_sequence[p_index]
                if priority_sequence
                else priority[(p_offset + p_index) % len(priority)]
            )
            p_index += 1
            schedule.append((spec, "official_target"))
        else:
            spec = diverse[(d_offset + d_index) % len(diverse)]
            d_index += 1
            schedule.append((spec, "diverse_public"))
    assert p_index == n_priority and d_index == n_diverse
    return tuple(schedule)


def _public_training_seat(
    *,
    seed: int,
    iteration: int,
    archetype: str,
    opponent_id: str,
    occurrence: int,
) -> int:
    """Alternate seats independently for every deck/opponent cell.

    Public-group interleaving can have the same parity as the global game
    index (the 50/50 specialist wave alternates official/diverse exactly).
    Using ``game_i % 2`` therefore pins an entire group to one seat.  A stable
    per-cell offset plus occurrence parity gives exact balance for even cells
    and at most one-game skew for odd cells without correlating seat to group.
    """
    token = (
        f"{int(seed)}:{int(iteration)}:{str(archetype)}:"
        f"{str(opponent_id)}:public_training_seat_v1"
    ).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1
    return int((offset + max(0, int(occurrence))) & 1)


def _paired_official_exploit(
    *,
    seed: int,
    iteration: int,
    archetype: str,
    opponent_id: str,
    occurrence: int,
    fraction: float,
) -> bool:
    """Select a deterministic fraction of whole two-seat occurrence pairs.

    Consecutive occurrences for a public deck/opponent cell use opposite seats.
    Making the temperature decision on ``occurrence // 2`` prevents behavior
    temperature from becoming a seat label.  The hashed phase spreads selected
    pairs across the wave while preserving the requested fraction to within one
    pair for any prefix.
    """
    frac = min(1.0, max(0.0, float(fraction)))
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    pair_index = max(0, int(occurrence)) // 2
    token = (
        f"{int(seed)}:{int(iteration)}:{str(archetype)}:"
        f"{str(opponent_id)}:official_exploit_pair_v1"
    ).encode("utf-8")
    phase = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 1009
    before = math.floor((phase + pair_index) * frac)
    after = math.floor((phase + pair_index + 1) * frac)
    return bool(after > before)


def _build_collect_jobs(
    *,
    n_games: int,
    ckpt: Path,
    digest: str,
    model_generation: int,
    decks: list[tuple[str, list[int]]],
    specs: list,
    seed: int,
    game_timeout_s: int,
    mode: str,
    collect_temperature: float = 1.0,
    max_context: Optional[int] = None,
    opponent_pool: Optional[list[Any]] = None,
    self_play_frac: Optional[float] = None,
    balanced_eval: bool = False,
    ladder_mix: Any = None,
    iteration: int = 0,
    priority_specs: Optional[list[Any]] = None,
    priority_frac: float = 0.0,
    priority_weights: Optional[dict[str, float]] = None,
    official_exploit_opponents: Optional[tuple[str, ...]] = None,
    official_exploit_frac: float = 0.0,
    official_exploit_temperature: float = 0.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(self_play_jobs, baseline_jobs)`` — self-play is the primary signal."""
    self_jobs: list[dict[str, Any]] = []
    base_jobs: list[dict[str, Any]] = []
    if not decks:
        return self_jobs, base_jobs
    ctx = int(max_context if max_context is not None else pure_rl_model_config().max_context)
    frac = float(
        self_play_frac
        if self_play_frac is not None
        else getattr(config.PURE_RL, "self_play_frac", 0.85)
    )
    frac = min(1.0, max(0.0, frac))
    n_self = int(round(n_games * frac))
    if n_games > 0 and frac > 0 and n_self == 0:
        n_self = 1
    has_public_specs = bool(specs or priority_specs)
    if n_games > 0 and frac < 1 and n_self == n_games and has_public_specs:
        n_self = max(0, n_games - 1)
    if not has_public_specs:
        n_self = n_games
    ladder_catalog: dict[str, tuple[str, list[int]]] = {}
    self_our_schedule: tuple[str, ...] = ()
    self_opp_schedule: tuple[str, ...] = ()
    public_our_schedule: tuple[str, ...] = ()
    public_spec_schedule: tuple[tuple[Any, str], ...] = ()
    public_seat_counts: dict[tuple[str, str], int] = {}
    exploit_opponents = frozenset(
        str(opponent_id) for opponent_id in (official_exploit_opponents or ())
    )
    ladder_wave_provenance: dict[str, Any] = {}
    if ladder_mix is not None:
        if balanced_eval:
            raise ValueError("ladder-weighted collection is not a formal eval sampler")
        ladder_catalog = {str(name): (str(name), list(deck)) for name, deck in decks}
        expected_ids = {entry.deck_id for entry in ladder_mix.decks}
        if set(ladder_catalog) != expected_ids:
            raise RuntimeError(
                "bound ladder catalog does not match the mix IDs: "
                f"missing={sorted(expected_ids - set(ladder_catalog))} "
                f"extra={sorted(set(ladder_catalog) - expected_ids)}"
            )
        self_our_schedule = ladder_mix.schedule_ids(
            n_self,
            seed=seed,
            iteration=iteration,
            stream="self_play_our",
        )
        raw_opp_schedule = ladder_mix.schedule_ids(
            n_self,
            seed=seed,
            iteration=iteration,
            stream="self_play_opp",
        )
        self_opp_schedule = _derange_deck_schedule(
            self_our_schedule, raw_opp_schedule
        )
        public_our_schedule = ladder_mix.schedule_ids(
            n_games - n_self,
            seed=seed,
            iteration=iteration,
            stream="public_our",
        )
        ladder_wave_provenance = {
            "mix_id": ladder_mix.mix_id,
            "artifact_sha256": ladder_mix.artifact_sha256,
            "basis": "train",
            "iteration": int(iteration),
            "self_play_our_quotas": ladder_mix.quotas(n_self),
            "self_play_opp_quotas": ladder_mix.quotas(n_self),
            "public_our_quotas": ladder_mix.quotas(n_games - n_self),
        }
    if n_games > n_self and float(priority_frac) > 0.0:
        public_spec_schedule = _interleaved_opponent_schedule(
            n_games - n_self,
            priority_specs=list(priority_specs or []),
            diverse_specs=list(specs),
            priority_frac=float(priority_frac),
            seed=int(seed),
            iteration=int(iteration),
            priority_weights=priority_weights,
        )
    # Portable baseline identity hashing walks each installed source tree.
    # A 64k public wave must do that once per opponent, not once per game.
    spec_payloads: dict[str, dict[str, Any]] = {}
    for spec in [*list(specs), *list(priority_specs or [])]:
        spec_id = str(spec.id)
        payload = _spec_payload(spec)
        previous = spec_payloads.setdefault(spec_id, payload)
        if previous != payload:
            raise RuntimeError(
                f"opponent ID {spec_id!r} resolves to conflicting payloads"
            )
    raw_pool = list(opponent_pool or [{"path": str(ckpt), "digest": str(digest)}])
    pool: list[dict[str, str]] = []
    for entry in raw_pool:
        if isinstance(entry, dict):
            opp_path = str(entry.get("path") or "")
            opp_digest = str(entry.get("digest") or "")
        elif hasattr(entry, "path") and hasattr(entry, "digest"):
            opp_path = str(entry.path)
            opp_digest = str(entry.digest)
        else:
            opp_path = str(entry)
            opp_digest = str(digest) if opp_path == str(ckpt) else ""
        if not opp_path or not opp_digest:
            raise RuntimeError(
                "self-play opponent pool requires path+digest identities; "
                f"invalid entry={entry!r}"
            )
        pool.append({"path": opp_path, "digest": opp_digest})
    if not pool:
        pool = [{"path": str(ckpt), "digest": str(digest)}]
    for game_i in range(n_games):
        if balanced_eval and specs:
            # Adjacent games are the same deck/opponent with candidate seat
            # flipped. This removes the old opponent↔seat parity confound.
            pair_i = game_i // 2
            spec_i = pair_i % len(specs)
            deck_i = (pair_i // len(specs)) % len(decks)
            arch, deck = decks[deck_i]
            our_seat = game_i % 2
        elif ladder_mix is not None:
            if game_i < n_self:
                scheduled_id = self_our_schedule[game_i]
            else:
                scheduled_id = public_our_schedule[game_i - n_self]
            arch, deck = ladder_catalog[scheduled_id]
            # The family schedule is already SHA-256 permuted. Alternating the
            # seat keeps global balance without pinning a family to one seat.
            our_seat = (game_i + int(seed)) % 2
        else:
            arch, deck = decks[game_i % len(decks)]
            # Seat changes after one complete deck rotation, so each deck is
            # observed from both seats instead of being pinned to index parity.
            our_seat = (game_i // len(decks)) % 2
        common = {
            "job_index": game_i,
            "checkpoint": str(ckpt),
            "checkpoint_digest": digest,
            "model_generation": model_generation,
            "model_max_context": ctx,
            "our_deck": list(deck),
            "our_seat": our_seat,
            "mcts_sims": 0,
            "mcts_move_time": 0.0,
            "game_timeout_s": int(game_timeout_s),
            "agent_mode": "policy",
            "sample_actions": True,
            "action_temperature": float(collect_temperature),
            "seed": int(seed + game_i),
            "device": "cpu",
            "training_eligible": True,
            "archetype": arch,
            "collect_privileged_belief": True,
        }
        if ladder_mix is not None:
            common["deck_mix"] = {
                **ladder_wave_provenance,
                "scheduled_our_deck_id": arch,
                "stream": (
                    "self_play_our" if game_i < n_self else "public_our"
                ),
            }
        if game_i < n_self or not has_public_specs:
            opp_identity = pool[game_i % len(pool)]
            opp_ckpt = opp_identity["path"]
            opp_digest = opp_identity["digest"]
            if ladder_mix is not None:
                opp_arch = self_opp_schedule[game_i]
                _opp_name, opp_deck = ladder_catalog[opp_arch]
            elif len(decks) > 1:
                our_i = game_i % len(decks)
                matchup_round = game_i // len(decks)
                opp_i = (
                    our_i + 1 + (matchup_round % (len(decks) - 1))
                ) % len(decks)
                opp_arch, opp_deck = decks[opp_i]
            else:
                opp_arch, opp_deck = arch, deck
            self_jobs.append(
                {
                    **common,
                    "opponent_checkpoint": opp_ckpt,
                    "opponent_checkpoint_digest": opp_digest,
                    "opponent_id": f"self:{Path(opp_ckpt).name}",
                    "opp_deck": list(opp_deck),
                    "opp_archetype": opp_arch,
                    "collect_both_seats": bool(str(opp_digest) == str(digest)),
                    "target_provenance": {
                        "pure_rl": True,
                        "soft_policy_targets": False,
                        "collect": "self_play",
                        "self_play": True,
                        "behavior_checkpoint": str(ckpt),
                        "behavior_checkpoint_digest": str(digest),
                        "opponent_checkpoint": str(opp_ckpt),
                        "opponent_checkpoint_digest": str(opp_digest),
                        "mcts_sims": 0,
                        "action_temperature": float(collect_temperature),
                        **(
                            {
                                "deck_mix": {
                                    **ladder_wave_provenance,
                                    "scheduled_our_deck_id": arch,
                                    "scheduled_opp_deck_id": opp_arch,
                                    "stream": "self_play",
                                }
                            }
                            if ladder_mix is not None
                            else {}
                        ),
                    },
                }
            )
        else:
            if balanced_eval:
                spec = specs[(game_i // 2) % len(specs)]
                opponent_training_group = "formal_eval"
            elif public_spec_schedule:
                spec, opponent_training_group = public_spec_schedule[
                    game_i - n_self
                ]
            else:
                spec = specs[(game_i - n_self) % len(specs)]
                opponent_training_group = "diverse_public"
            if not balanced_eval:
                seat_key = (str(arch), str(spec.id))
                occurrence = int(public_seat_counts.get(seat_key, 0))
                common["our_seat"] = _public_training_seat(
                    seed=int(seed),
                    iteration=int(iteration),
                    archetype=str(arch),
                    opponent_id=str(spec.id),
                    occurrence=occurrence,
                )
                public_seat_counts[seat_key] = occurrence + 1
                sharpened = bool(
                    opponent_training_group == "official_target"
                    and str(spec.id) in exploit_opponents
                    and _paired_official_exploit(
                        seed=int(seed),
                        iteration=int(iteration),
                        archetype=str(arch),
                        opponent_id=str(spec.id),
                        occurrence=occurrence,
                        fraction=float(official_exploit_frac),
                    )
                )
            else:
                sharpened = False
            behavior_temperature = (
                float(official_exploit_temperature)
                if sharpened
                else float(collect_temperature)
            )
            common["action_temperature"] = behavior_temperature
            base_jobs.append(
                {
                    **common,
                    "spec": dict(spec_payloads[str(spec.id)]),
                    "require_portable_baseline_contract": True,
                    "opponent_id": spec.id,
                    "target_provenance": {
                        "pure_rl": True,
                        "soft_policy_targets": False,
                        "collect": "public_mix",
                        "opponent_training_group": opponent_training_group,
                        "opponent_sampling_weight": (
                            float(priority_weights[str(spec.id)])
                            if opponent_training_group == "official_target"
                            and priority_weights is not None
                            else None
                        ),
                        "opponent_schedule": (
                            "adaptive_exact_heldout_gap_v1"
                            if opponent_training_group == "official_target"
                            and priority_weights is not None
                            else "uniform_round_robin_v1"
                        ),
                        "self_play": False,
                        "behavior_checkpoint": str(ckpt),
                        "behavior_checkpoint_digest": str(digest),
                        "mcts_sims": 0,
                        "action_temperature": behavior_temperature,
                        "behavior_mode": (
                            "official_exploit_sharpened_v1"
                            if sharpened
                            else "base_sampling_temperature_v1"
                        ),
                        "seat_schedule": (
                            "formal_eval_paired_v1"
                            if balanced_eval
                            else "per_opponent_archetype_alternating_v1"
                        ),
                        **(
                            {
                                "deck_mix": {
                                    **ladder_wave_provenance,
                                    "scheduled_our_deck_id": arch,
                                    "stream": "public_our",
                                }
                            }
                            if ladder_mix is not None
                            else {}
                        ),
                    },
                }
            )
    return self_jobs, base_jobs


def _dataset_from_replay_window(
    run_dir: Path,
    it: int,
    *,
    initial_replay_shards: Optional[list[Path]] = None,
) -> Any:
    """Fresh-data bias (Spinning Up on-policy theme): short window only.

    ``bootstrap_mix`` must stay 0 — refuse soft CE / starter replay contamination.
    """
    from poke_bot.dataset import BootstrapDataset

    mix = float(getattr(config.PURE_RL, "bootstrap_mix", 0.0))
    if mix > 0.0:
        raise SystemExit(
            f"PURE_RL fail-closed: bootstrap_mix={mix} > 0 "
            "(Spinning Up / Abhyuday: no starter CE clone; fresh AWR data only)"
        )
    window = max(1, int(getattr(config.PURE_RL, "replay_window_shards", 2)))
    seqs = []
    # A clean lineage handoff may carry the immediately preceding immutable
    # shard without copying 3+ GiB or pretending it belongs to local iter 0.
    # It participates only in the first update; iter 1 naturally has two local
    # shards and restores the ordinary rolling-window rule.
    if int(it) == 0 and window > 1:
        for shard in list(initial_replay_shards or [])[-(window - 1) :]:
            ds = dataset_from_shard(
                Path(shard),
                verify_info_set=False,
                max_context=pure_rl_model_config().max_context,
            )
            seqs.extend(list(ds.sequences))
    for j in range(max(0, it - window + 1), it + 1):
        shard = run_dir / "shards" / f"iter_{j:05d}.jsonl"
        if not shard.is_file():
            continue
        ds = dataset_from_shard(
            shard,
            verify_info_set=False,
            max_context=pure_rl_model_config().max_context,
        )
        seqs.extend(list(ds.sequences))
    return BootstrapDataset(sequences=seqs)


def _consume_results(
    results_iter,
    writer: CompactShardWriter,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    progress: Optional[_TqdmProgress] = None,
    live_wr_gate: Optional[tuple[float, int]] = None,
    replay_cache: Optional[StreamingReplayCache] = None,
    required_checkpoint_digest: Optional[str] = None,
    live_wr_opponent_ids: Optional[tuple[str, ...]] = None,
) -> None:
    """``live_wr_gate=(target_wr, min_games)`` streams a running WR onto the
    bar as heldout games land (official baselines only) — passed only from
    ``_heldout_eval`` so regular collect waves don't mislabel a practice
    mix's partial WR as the gate signal.
    """
    wr_wins = 0.0
    wr_games = 0
    for res in results_iter:
        heldout_contract_invalid = bool(
            required_checkpoint_digest is not None
            and (
                str(res.get("checkpoint_digest") or "")
                != str(required_checkpoint_digest)
                or str(res.get("action_selection") or "") != "greedy"
            )
        )
        rows.append(
            {
                "opponent_id": res.get("opponent_id"),
                "our_seat": res.get("our_seat"),
                "winner": res.get("winner"),
                "baseline_failed": bool(res.get("baseline_failed")),
                "our_failed": bool(res.get("our_failed")),
                "self_play": bool(res.get("self_play")),
                "leaf_self_play_mode": res.get("leaf_self_play_mode"),
                "leaf_remote": bool(res.get("leaf_remote")),
                "multi_env": bool(res.get("multi_env")),
                "job_index": res.get("job_index"),
                "archetype": res.get("archetype"),
                "checkpoint_digest": res.get("checkpoint_digest"),
                "action_selection": res.get("action_selection"),
                "error": res.get("error"),
                "invalid": bool(
                    res.get("our_failed")
                    or res.get("resource_error")
                    or res.get("cancelled")
                    or res.get("trust_failure")
                    or res.get("game_timeout")
                    or heldout_contract_invalid
                ),
                "heldout_contract_invalid": heldout_contract_invalid,
            }
        )
        if live_wr_gate is not None and progress is not None:
            opp = str(res.get("opponent_id") or "")
            counted_ids = set(live_wr_opponent_ids or OFFICIAL_BASELINE_IDS)
            counted = bool(
                opp in counted_ids
                and not res.get("baseline_failed")
                and not res.get("our_failed")
                and not res.get("resource_error")
                and not res.get("cancelled")
                and not res.get("trust_failure")
                and not res.get("game_timeout")
                and not heldout_contract_invalid
                and res.get("winner") is not None
            )
            if counted:
                winner = int(res["winner"])
                our_seat = int(res.get("our_seat") or 0)
                wr_games += 1
                if winner == 2:
                    wr_wins += 0.5
                elif winner == our_seat:
                    wr_wins += 1.0
                target_wr, _min_games = live_wr_gate
                progress.set_wr(
                    (wr_wins / wr_games) if wr_games else 0.0,
                    wr_games,
                    target=target_wr,
                )
        mode = str(res.get("leaf_self_play_mode") or "")
        if mode:
            leaf_modes = stats.setdefault("leaf_modes", {})
            leaf_modes[mode] = int(leaf_modes.get(mode, 0)) + 1
        if res.get("leaf_remote"):
            stats["leaf_remote"] = int(stats.get("leaf_remote", 0)) + 1
        if res.get("multi_env"):
            stats["multi_env_games"] = int(stats.get("multi_env_games", 0)) + 1
        if res.get("baseline_failed"):
            stats["baseline_failed"] += 1
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        if res.get("our_failed") or res.get("resource_error") or res.get("cancelled"):
            if res.get("our_failed"):
                stats["our_failed"] += 1
            if res.get("resource_error"):
                stats["resource_error"] += 1
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        stats["ok"] += 1
        if res.get("self_play"):
            stats["self_play"] = int(stats.get("self_play", 0)) + 1
        encoded_records = list(res.get("record_jsons") or [])
        if not encoded_records and res.get("record_json"):
            encoded_records = [res["record_json"]]
        records = []
        for encoded in encoded_records:
            try:
                parsed = json.loads(encoded) if isinstance(encoded, str) else encoded
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                records.append(parsed)
        if not records:
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        written = 0
        for record in records:
            game = _record_to_compact_game(record)
            if game is None:
                continue
            writer.write_game(game)
            if replay_cache is not None:
                replay_cache.note_append()
            written += 1
        if written <= 0:
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        stats["with_record"] += 1
        stats["trajectories_written"] = int(
            stats.get("trajectories_written", 0)
        ) + written
        if progress is not None:
            progress.tick(decisions=writer.n_decisions)


def _flatten_batch_results(results_iter):
    """Expand multi-env worker outputs (list[dict]) into a flat result stream."""
    for item in results_iter:
        if isinstance(item, list):
            for res in item:
                yield res
        else:
            yield item


class BetweenIterSyncError(RuntimeError):
    """Between-iter weight sync hard-gate failed (digest mismatch / incomplete)."""


def _hard_gate_publish_weights(
    *,
    leaf: "_LeafFarm",
    remote_farm: Any,
    ckpt: Path,
    digest: str,
    version: int,
    required_endpoints: list[str],
    reload_local: bool = True,
) -> dict[str, Any]:
    """Hard-gate local+remote weight publish before next-iter work continues.

    Fail-closed: every local leaf ack and every required remote reload/pin must
    report ``checkpoint_digest == digest``. Soft WARN-and-continue is forbidden
    at this boundary — workers must not dispatch on mixed digests.
    """
    from poke_bot.remote_jobs import RemoteJobsError, parse_endpoint

    proof: dict[str, Any] = {
        "digest": digest,
        "version": int(version),
        "checkpoint": str(ckpt),
        "local_ok": False,
        "remote_ok": False,
        "remote_endpoints": [],
    }
    dig = str(digest)
    if not dig.startswith("sha256:"):
        raise BetweenIterSyncError(f"invalid publish digest: {dig!r}")

    # 1) Local GPU leaves — reload already hard-fails on bad/mismatched acks.
    if leaf.remote_channel is not None:
        already = (
            not reload_local
            and leaf.digest == dig
            and leaf.remote_channel.get("expected_digest") == dig
        )
        if already:
            proof["local_reload_skipped"] = True
        else:
            leaf.reload(ckpt, dig)
        if leaf.digest != dig:
            raise BetweenIterSyncError(
                f"local leaf digest not published: have={leaf.digest} want={dig}"
            )
        ch = leaf.remote_channel
        if ch.get("expected_digest") != dig:
            raise BetweenIterSyncError(
                "local remote_channel expected_digest not updated after reload"
            )
        ch["generation"] = int(version)
        ch["expected_version"] = int(leaf.version)
        proof["local_ok"] = True
        proof["local_version"] = int(leaf.version)
    else:
        proof["local_ok"] = True  # cpu leaf-eval path
        proof["local_skipped"] = True

    # 2) Remotes — require every configured endpoint; no soft-drop at boundary.
    req = [e.strip() for e in required_endpoints if str(e).strip()]
    if not req:
        proof["remote_ok"] = True
        proof["remote_skipped"] = True
        print(
            f"[pure_rl] BETWEEN_ITER_HARD_GATE ok local digest={dig[:19]}… "
            f"version={version} remotes=none",
            flush=True,
        )
        return proof
    if remote_farm is None:
        raise BetweenIterSyncError(
            "remote farm missing at between-iter hard-gate "
            f"(required={req})"
        )

    # Re-attach any soft-dropped clients before the strict sync.
    try:
        remote_farm._reconnect_missing()
    except Exception as exc:  # noqa: BLE001
        raise BetweenIterSyncError(
            f"remote reconnect before hard-gate failed: {exc}"
        ) from exc

    alive = {(c.host, int(c.port)): c for c in remote_farm.clients}
    missing: list[str] = []
    clients_ordered = []
    for ep in req:
        host, port = parse_endpoint(ep)
        key = (host, int(port))
        client = alive.get(key)
        if client is None:
            missing.append(ep)
        else:
            clients_ordered.append((ep, client))
    if missing:
        raise BetweenIterSyncError(
            "between-iter hard-gate: required remotes not connected: "
            + ", ".join(missing)
        )

    remote_proof: list[dict[str, Any]] = []
    errors: list[str] = []
    for ep, client in clients_ordered:
        try:
            # A farm client may still be present in ``clients`` after its
            # worker was replaced or recycled.  ``_reconnect_missing`` only
            # recreates absent entries, so explicitly refresh every control
            # socket at this strict boundary before issuing reload/pin.  This
            # is intentionally fail-closed: if a required endpoint cannot
            # reconnect, no next-iteration work may start on mixed weights.
            client.reconnect()
            reload_reply = client.reload_checkpoint(
                str(ckpt), digest=dig, version=int(version)
            )
            got = reload_reply.get("checkpoint_digest")
            if not reload_reply.get("ok", False) or got != dig:
                raise RemoteJobsError(
                    f"reload digest mismatch on {ep}: reply={reload_reply!r}"
                )
            pin_reply = client.pin_checkpoint(str(ckpt), digest=dig)
            pin_got = pin_reply.get("checkpoint_digest")
            if not pin_reply.get("ok", False) or pin_got != dig:
                raise RemoteJobsError(
                    f"pin digest mismatch on {ep}: reply={pin_reply!r}"
                )
            # Re-hello confirms the worker's published primary digest.
            info = client.reconnect()
            hello_dig = getattr(info, "checkpoint_digest", None)
            if hello_dig != dig:
                raise RemoteJobsError(
                    f"hello digest mismatch on {ep}: expected {dig}, got {hello_dig}"
                )
            remote_proof.append(
                {
                    "endpoint": ep,
                    "reload_digest": got,
                    "pin_digest": pin_got,
                    "hello_digest": hello_dig,
                    "version": reload_reply.get("version"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ep}: {exc}")
    if errors:
        raise BetweenIterSyncError(
            "between-iter hard-gate remote sync failed: " + "; ".join(errors)
        )
    proof["remote_ok"] = True
    proof["remote_endpoints"] = remote_proof
    print(
        f"[pure_rl] BETWEEN_ITER_HARD_GATE ok digest={dig[:19]}… "
        f"version={version} local=1 remotes={len(remote_proof)} "
        f"endpoints={[r['endpoint'] for r in remote_proof]}",
        flush=True,
    )
    return proof


def _remote_dispatch_slots(
    *,
    remote_farm: Any,
    scheduler: Any,
    baseline_workers: int,
    kind: str = "self_play",
    allow_remote_play: bool = False,
) -> tuple[int, int, str]:
    """Return (local_slots, remote_cap, weight_bits) for additive collect.

    ``kind="play"`` is the public-mix wave (``baseline_jobs`` vs the public /
    recent-self roster). It finishes fast locally, so by default it is
    routed local-only (``PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1``), keeping remotes
    free for the self-play wave and other/slower work. Self-play
    (``kind="self_play"``) always keeps the original shared
    ``PURE_RL_REBALANCE_MIN_LOCAL_FRAC`` behavior, unaffected.
    """
    from poke_bot.remote_jobs import describe_endpoint_weights, weighted_remote_capacity

    assert remote_farm is not None
    is_public_mix = kind == "play"
    if is_public_mix and _public_mix_local_only() and not allow_remote_play:
        local_slots = max(1, int(baseline_workers))
        weight_bits = (
            "local_only (PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1; remotes free for "
            "self_play/other work)"
        )
        return local_slots, 0, weight_bits
    demand = None
    if scheduler is not None:
        try:
            scheduler.bind_remote_endpoints(remote_farm.clients)
            demand = scheduler.remote_demand()
        except Exception:
            demand = None
    weight_rows = describe_endpoint_weights(
        remote_farm.clients, demand_by_endpoint=demand
    )
    remote_cap = max(
        1,
        weighted_remote_capacity(remote_farm.clients, demand_by_endpoint=demand),
    )
    weight_bits = ", ".join(
        f"{r['endpoint']} demand={r.get('demand_workers', '?')}"
        f"→slots={r['dispatch_slots']}/{r.get('capacity_workers', r['advertised_workers'])}"
        f"(default={r.get('default_workers', '?')} max={r.get('max_workers', '?')})"
        for r in weight_rows
    )
    # Additive: keep full local baseline (≈96) and pile remotes on top.
    # Do NOT halve local_slots for RR — that stole from local until remotes
    # filled, capping the pie at ~baseline instead of local+remote.
    if is_public_mix and allow_remote_play:
        # Formal held-out is deliberately additive: keep the proven local pool
        # full and add compatible remote sockets.  The 95% public-mix bias would
        # otherwise manufacture hundreds of local slots and exceed leaf queues.
        local_slots = max(1, int(baseline_workers))
        weight_bits = "formal_heldout_additive; " + weight_bits
        return local_slots, remote_cap, weight_bits
    if is_public_mix:
        # Heavy-local fallback (LOCAL_ONLY explicitly disabled): bias much
        # harder toward local than the self-play share below.
        min_local_frac = _public_mix_min_local_frac()
    else:
        min_local_frac = float(
            os.environ.get("PURE_RL_REBALANCE_MIN_LOCAL_FRAC", "0.40")
        )
    min_local_frac = min(0.95, max(0.05, min_local_frac))
    min_local_slots = max(
        1,
        int(math.ceil(remote_cap * min_local_frac / (1.0 - min_local_frac))),
    )
    local_slots = max(min_local_slots, max(1, int(baseline_workers)))
    return local_slots, remote_cap, weight_bits


def _collect_wave(
    *,
    self_play_jobs: list[dict[str, Any]],
    baseline_jobs: list[dict[str, Any]],
    shard_path: Path,
    n_workers: int,
    leaf_channel,
    remote_farm,
    worker_play,
    worker_self_play,
    multi_env_per_worker: int = 1,
    worker_self_play_multi=None,
    iteration: int = 0,
    stage_label: Optional[str] = None,
    live_wr_gate: Optional[tuple[float, int]] = None,
    allow_remote_play: bool = False,
    required_checkpoint_digest: Optional[str] = None,
    live_wr_opponent_ids: Optional[tuple[str, ...]] = None,
) -> tuple[CompactShardWriter, list[dict[str, Any]], dict[str, Any]]:
    """Self-play + public mix with additive LAN remotes (local-primary).

    ``stage_label`` overrides the tqdm bar's default ``collect:public_mix``
    name (e.g. ``"heldout"`` so the heldout gate wave doesn't look like a
    regular training collect). ``live_wr_gate`` streams a running WR onto
    that bar for the baseline_jobs wave only — see ``_heldout_eval``.
    """
    from poke_bot.pure_rl.multi_env_self_play import (
        chunk_jobs,
        process_worker_count,
    )
    from poke_bot.pure_rl.mid_iter_scheduler import (
        MidIterScheduler,
        log_scheduler_banner,
        mid_iter_scheduler_enabled,
    )
    from poke_bot.remote_jobs import (
        iter_additive_results,
        iter_scheduled_additive_results,
        weighted_remote_capacity,
    )
    from poke_bot.worker_pool import WorkerPool

    writer = CompactShardWriter(shard_path)
    replay_cache: Optional[StreamingReplayCache] = None
    rows: list[dict[str, Any]] = []
    multi_n = max(1, int(multi_env_per_worker))
    proc_workers = process_worker_count(n_workers, multi_n)
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "n_self_play_jobs": len(self_play_jobs),
        "n_baseline_jobs": len(baseline_jobs),
        "multi_env_per_worker": multi_n,
        "proc_workers": proc_workers,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
        "execution_origin_counts": {},
        "remote_endpoint_counts": {},
        "remote_self_play_endpoint_counts": {},
    }
    if not self_play_jobs and not baseline_jobs:
        return writer, rows, stats
    replay_cache = StreamingReplayCache.maybe_start(
        shard_path,
        verify_info_set=False,
        max_context=pure_rl_model_config().max_context,
    )

    # Hello advertised defaults — bar uses demand-based dispatch slots after bind
    # (must track live open sockets, not frozen hello workers).
    remotes_hello = int(remote_farm.total_workers) if remote_farm is not None else 0
    use_remotes = bool(remote_farm is not None and remotes_hello > 0)
    # Multi-env batching only when remotes are down. With remotes up we dispatch
    # single-game jobs so the LAN farm can share the same list — use full
    # sim_workers fan-out so sticky leaf binds cover the striped GPU0/GPU1 map
    # (proc_workers=24 would pin every client onto the first 24 GPU1 servers).
    use_multi_env_batches = bool(
        (not use_remotes) and multi_n > 1 and worker_self_play_multi is not None
    )
    local_workers = (
        max(1, int(proc_workers))
        if use_multi_env_batches
        else max(1, int(n_workers))
    )
    # Fail-closed guard: WorkerPool requires one resp_q per worker. Never open
    # a pool larger than the leaf farm's queue list (live_pool can bump workers
    # while leaf rebuild is deferred).
    if isinstance(leaf_channel, dict):
        n_qs = len(leaf_channel.get("resp_qs") or [])
        if n_qs > 0 and local_workers > n_qs:
            print(
                f"[pure_rl] WARN clamp local_workers {local_workers}->{n_qs} "
                f"to match leaf resp_qs (avoid queues<workers crash)",
                flush=True,
            )
            local_workers = n_qs
    use_scheduler = bool(use_remotes and mid_iter_scheduler_enabled())
    scheduler = (
        MidIterScheduler.from_env(baseline_workers=max(1, int(n_workers)))
        if use_scheduler
        else None
    )
    if scheduler is not None and remote_farm is not None:
        try:
            scheduler.bind_remote_endpoints(remote_farm.clients)
        except Exception as exc:
            print(f"[pure_rl] bind_remote_endpoints failed: {exc!r}", flush=True)
    remotes = remotes_hello
    if use_remotes and remote_farm is not None:
        try:
            demand0 = scheduler.remote_demand() if scheduler is not None else None
            remotes = max(
                1,
                weighted_remote_capacity(
                    remote_farm.clients, demand_by_endpoint=demand0
                ),
            )
        except Exception:
            remotes = remotes_hello
    if scheduler is not None:
        log_scheduler_banner(scheduler)
    print(
        f"[pure_rl] self_play_pool workers={local_workers} "
        f"multi_env_batches={int(use_multi_env_batches)} "
        f"proc_cap={proc_workers} sim_workers={n_workers} remotes={remotes} "
        f"(hello_defaults={remotes_hello}; bar tracks live dispatch sockets)",
        flush=True,
    )

    execution_lock = threading.Lock()

    def _record_execution(info: dict[str, Any]) -> None:
        origin = str(info.get("origin") or "unknown")
        kind = str(info.get("kind") or "unknown")
        endpoint = str(info.get("endpoint") or "")
        with execution_lock:
            origins = stats["execution_origin_counts"]
            origins[origin] = int(origins.get(origin, 0)) + 1
            if origin == "remote" and endpoint:
                endpoints_seen = stats["remote_endpoint_counts"]
                endpoints_seen[endpoint] = int(endpoints_seen.get(endpoint, 0)) + 1
                if kind == "self_play":
                    self_play_seen = stats["remote_self_play_endpoint_counts"]
                    self_play_seen[endpoint] = int(self_play_seen.get(endpoint, 0)) + 1

    def _additive_iter(
        *,
        pool,
        local_fn,
        jobs,
        kind: str,
        baseline_workers: int,
        progress: Optional[_TqdmProgress] = None,
    ):
        local_slots, remote_cap, weight_bits = _remote_dispatch_slots(
            remote_farm=remote_farm,
            scheduler=scheduler,
            baseline_workers=baseline_workers,
            kind=kind,
            allow_remote_play=allow_remote_play,
        )
        public_mix_local_only = bool(kind == "play" and remote_cap == 0)
        total_slots = local_slots + remote_cap
        share = local_slots / float(total_slots)
        remote_max_total = 0
        if not public_mix_local_only:
            try:
                remote_max_total = int(
                    sum(
                        int(scheduler.remote_maxima.get(c.endpoint, 0))
                        for c in remote_farm.clients
                    )
                    if scheduler is not None
                    else 0
                )
            except Exception:
                remote_max_total = 0
        max_total = local_slots + max(remote_cap, remote_max_total)
        dem = remote_cap
        if scheduler is not None and not public_mix_local_only:
            try:
                dem = int(sum(scheduler.remote_demand().values()))
            except Exception:
                dem = remote_cap
        if progress is not None:
            progress.set_remotes(remote_cap, demand=dem)
        print(
            f"[pure_rl] {kind} remote_weights {weight_bits} "
            f"local_slots={local_slots} remote_slots={remote_cap} "
            f"total_slots={total_slots} max_total≈{max_total} "
            f"local_share={share:.0%} "
            f"(additive kind={kind}; local stays full, remotes on top; "
            f"bar remotes=live_sockets start={remote_cap}"
            + (
                "; mid_iter_scheduler=on, remote_chunked_rtt)"
                if scheduler is not None
                else ")"
            ),
            flush=True,
        )

        def _on_remote_slots(info: dict[str, int]) -> None:
            if progress is None:
                return
            try:
                progress.set_remotes(
                    int(info.get("active", 0)),
                    demand=int(info["demand"]) if "demand" in info else None,
                )
            except Exception:
                pass

        if scheduler is not None:
            return iter_scheduled_additive_results(
                local_pool=pool,
                local_fn=local_fn,
                jobs=jobs,
                remote_clients=remote_farm.clients,
                kind=kind,
                scheduler=scheduler,
                local_workers=local_slots,
                remote_workers=remote_cap,
                on_remote_slots=_on_remote_slots,
                on_execution=_record_execution,
            )
        return iter_additive_results(
            local_pool=pool,
            local_fn=local_fn,
            jobs=jobs,
            remote_clients=remote_farm.clients,
            kind=kind,
            local_workers=local_slots,
            remote_workers=remote_cap,
            on_execution=_record_execution,
        )

    # Primary: pure self-play — local MultiEnv + additive remote self_play sockets.
    if self_play_jobs:
        progress = _TqdmProgress(
            stage="collect:self_play",
            iteration=int(iteration),
            total=len(self_play_jobs),
            remotes=remotes,
        )
        try:
            with WorkerPool(
                num_workers=local_workers, remote_channel=leaf_channel
            ) as pool:
                if use_remotes:
                    # Single-game local fn so remotes can share the same job list.
                    # MultiEnv stays available when remotes are down.
                    results_iter = _additive_iter(
                        pool=pool,
                        local_fn=worker_self_play,
                        jobs=self_play_jobs,
                        kind="self_play",
                        baseline_workers=local_workers,
                        progress=progress,
                    )
                elif use_multi_env_batches:
                    batches = [
                        {"jobs": chunk}
                        for chunk in chunk_jobs(self_play_jobs, multi_n)
                    ]
                    results_iter = _flatten_batch_results(
                        pool.imap_unordered(worker_self_play_multi, batches)
                    )
                else:
                    results_iter = pool.imap_unordered(
                        worker_self_play, self_play_jobs
                    )
                _consume_results(
                    results_iter,
                    writer,
                    rows,
                    stats,
                    progress=progress,
                    replay_cache=replay_cache,
                )
        finally:
            progress.close()
    # Light mixture: public/roster via remotes (and local if no remotes).
    # Baseline path stays one-game-per-process (baselines use cg.game singleton).
    baseline_workers = max(1, int(n_workers))
    if baseline_jobs:
        progress = _TqdmProgress(
            stage=stage_label or "collect:public_mix",
            iteration=int(iteration),
            total=len(baseline_jobs),
            remotes=remotes,
        )
        try:
            with WorkerPool(
                num_workers=baseline_workers, remote_channel=leaf_channel
            ) as pool:
                if use_remotes:
                    results_iter = _additive_iter(
                        pool=pool,
                        local_fn=worker_play,
                        jobs=baseline_jobs,
                        kind="play",
                        baseline_workers=baseline_workers,
                        progress=progress,
                    )
                else:
                    results_iter = pool.imap_unordered(worker_play, baseline_jobs)
                _consume_results(
                    results_iter,
                    writer,
                    rows,
                    stats,
                    progress=progress,
                    live_wr_gate=live_wr_gate,
                    replay_cache=replay_cache,
                    required_checkpoint_digest=required_checkpoint_digest,
                    live_wr_opponent_ids=live_wr_opponent_ids,
                )
        finally:
            progress.close()
    strict_remote_execution = str(
        os.environ.get("POKEBOT_REMOTE_REQUIRE_ALL", "0")
    ).strip().lower() in ("1", "true", "yes", "on") or str(
        os.environ.get("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "0")
    ).strip().lower() in ("1", "true", "yes", "on")
    if self_play_jobs and use_remotes:
        expected_endpoints = sorted(
            {str(client.endpoint) for client in remote_farm.clients}
        )
        self_play_seen = dict(stats["remote_self_play_endpoint_counts"])
        missing_endpoints = [
            endpoint
            for endpoint in expected_endpoints
            if int(self_play_seen.get(endpoint, 0)) <= 0
        ]
        fallback_count = int(
            stats["execution_origin_counts"].get("local_fallback", 0)
        )
        stats["remote_execution_proof"] = {
            "strict": strict_remote_execution,
            "expected_endpoints": expected_endpoints,
            "self_play_completions_by_endpoint": self_play_seen,
            "missing_endpoints": missing_endpoints,
            "local_fallback_count": fallback_count,
        }
        if strict_remote_execution and (missing_endpoints or fallback_count):
            if replay_cache is not None:
                replay_cache.abort()
            raise RuntimeError(
                "strict remote execution proof failed: "
                f"missing_self_play_endpoints={missing_endpoints} "
                f"local_fallback_count={fallback_count}"
            )
    if replay_cache is not None:
        # A cache failure is optimization-only: finish abandons its staging
        # directory and the normal post-collect builder retries fail-closed.
        replay_cache.finish()
    return writer, rows, stats


def _remote_heldout_capability_audit(
    remote_farm: Any,
    *,
    required_endpoints: Optional[list[str]] = None,
) -> dict[str, Any]:
    required = {
        "greedy_play_v1",
        "active_checkpoint_job_barrier_v1",
        "play_result_contract_v1",
        "portable_baseline_spec_v1",
    }
    endpoints: list[dict[str, Any]] = []
    for client in list(getattr(remote_farm, "clients", ()) or ()):
        info = getattr(client, "info", None)
        capabilities = set(getattr(info, "capabilities", ()) or ())
        kinds = set(getattr(info, "job_kinds", ()) or ())
        missing = sorted(required - capabilities)
        if "play" not in kinds:
            missing.append("job_kind:play")
        endpoints.append(
            {
                "endpoint": str(getattr(client, "endpoint", "unknown")),
                "capabilities": sorted(capabilities),
                "missing": missing,
                "ok": not missing,
            }
        )
    connected = {str(row["endpoint"]) for row in endpoints}
    required_endpoints = [
        str(value).strip() for value in (required_endpoints or ()) if str(value).strip()
    ]
    missing_endpoints = sorted(set(required_endpoints) - connected)
    return {
        "required": sorted(required),
        "endpoints": endpoints,
        "required_endpoints": required_endpoints,
        "missing_endpoints": missing_endpoints,
        "passed": (
            bool(endpoints)
            and not missing_endpoints
            and all(row["ok"] for row in endpoints)
        ),
    }


def _audit_heldout_rows(
    rows: list[dict[str, Any]],
    *,
    n_games: int,
    checkpoint_digest: str,
    opponent_ids: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    from collections import Counter

    ordered_opponents = tuple(opponent_ids or OFFICIAL_BASELINE_IDS)
    if not ordered_opponents or len(set(ordered_opponents)) != len(ordered_opponents):
        raise ValueError("heldout opponent IDs must be unique and non-empty")
    if int(n_games) % (2 * len(ordered_opponents)):
        raise ValueError(
            "heldout games must divide evenly across opponents and both seats"
        )

    expected_ids = set(range(int(n_games)))
    counts = Counter(
        int(row["job_index"])
        for row in rows
        if isinstance(row.get("job_index"), int)
    )
    duplicates = sorted(job_id for job_id, count in counts.items() if count != 1)
    missing = sorted(expected_ids - set(counts))
    unexpected = sorted(set(counts) - expected_ids)
    for row in rows:
        job_id = row.get("job_index")
        if not isinstance(job_id, int) or counts.get(job_id, 0) != 1 or job_id not in expected_ids:
            row["invalid"] = True
            row["heldout_contract_invalid"] = True

    valid = [row for row in rows if not bool(row.get("invalid"))]
    per_opponent: dict[str, dict[str, int]] = {}
    for opponent in ordered_opponents:
        selected = [row for row in valid if row.get("opponent_id") == opponent]
        per_opponent[opponent] = {
            "games": len(selected),
            "seat0": sum(int(row.get("our_seat") == 0) for row in selected),
            "seat1": sum(int(row.get("our_seat") == 1) for row in selected),
        }
    expected_per_opponent = int(n_games) // len(ordered_opponents)
    exact_distribution = all(
        cell["games"] == expected_per_opponent
        and cell["seat0"] == expected_per_opponent // 2
        and cell["seat1"] == expected_per_opponent // 2
        for cell in per_opponent.values()
    )
    exact_weights = all(
        str(row.get("checkpoint_digest") or "") == str(checkpoint_digest)
        and str(row.get("action_selection") or "") == "greedy"
        for row in rows
    )
    passed = bool(
        len(rows) == int(n_games)
        and len(valid) == int(n_games)
        and not duplicates
        and not missing
        and not unexpected
        and exact_distribution
        and exact_weights
    )
    return {
        "passed": passed,
        "requested_games": int(n_games),
        "rows": len(rows),
        "valid_games": len(valid),
        "checkpoint_digest": str(checkpoint_digest),
        "greedy_required": True,
        "duplicate_job_ids": duplicates,
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "per_opponent": per_opponent,
        "expected_games_per_opponent": expected_per_opponent,
        "exact_distribution": exact_distribution,
        "exact_weights": exact_weights,
    }


def _heldout_eval(
    *,
    ckpt: Path,
    digest: str,
    n_games: int,
    decks: list[tuple[str, list[int]]],
    official_specs: list,
    seed: int,
    game_timeout_s: int,
    n_workers: int,
    leaf_channel,
    remote_farm,
    worker_play,
    worker_self_play,
    mode: str,
    allow_remote_play: bool = False,
    iteration: int = 0,
    gate_wr: float = 0.70,
    opponent_ids: Optional[tuple[str, ...]] = None,
    stage_label: str = "heldout",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy gate against official bots on a disjoint, balanced seed schedule.

    Streams a live win-rate readout onto the tqdm bar as games land, and
    labels the bar ``heldout`` (not ``collect:public_mix``) so it reads
    clearly as the gate check rather than a regular training collect wave.
    Specialist target training may use the same public opponent policies, but
    never these greedy jobs or their seed range.
    """
    _self_jobs, base_jobs = _build_collect_jobs(
        n_games=n_games,
        ckpt=ckpt,
        digest=digest,
        model_generation=0,
        decks=decks,
        specs=official_specs,
        seed=seed,
        game_timeout_s=game_timeout_s,
        mode=mode,
        self_play_frac=0.0,
        balanced_eval=True,
    )
    for job in base_jobs:
        job["training_eligible"] = False
        job["agent_mode"] = "policy"
        job["sample_actions"] = False
        job["greedy"] = True
        job["action_temperature"] = 1.0
    shard = paths.OUTPUTS_DIR / "pure_rl" / "_heldout_tmp" / f"{os.getpid()}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    try:
        _writer, rows, _stats = _collect_wave(
            self_play_jobs=[],
            baseline_jobs=base_jobs,
            shard_path=shard,
            n_workers=n_workers,
            leaf_channel=leaf_channel,
            remote_farm=remote_farm,
            worker_play=worker_play,
            worker_self_play=worker_self_play,
            iteration=int(iteration),
            stage_label=str(stage_label),
            live_wr_gate=(float(gate_wr), int(n_games)),
            allow_remote_play=bool(allow_remote_play),
            required_checkpoint_digest=str(digest),
            live_wr_opponent_ids=opponent_ids,
        )
    finally:
        try:
            shard.unlink(missing_ok=True)
        except Exception:
            pass
    audit = _audit_heldout_rows(
        rows,
        n_games=int(n_games),
        checkpoint_digest=str(digest),
        opponent_ids=opponent_ids,
    )
    return rows, audit


def _promotion_eval(
    *,
    candidate,
    incumbent,
    decks: list[tuple[str, list[int]]],
    n_games: int,
    n_workers: int,
    threshold: float,
    confidence: float,
    bootstrap_resamples: int,
    seed: int,
    game_timeout_s: int,
    model_generation: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seat/deck-balanced candidate-vs-incumbent non-regression gate."""
    from poke_bot.promotion import PromotionGateConfig, evaluate_candidate_gate
    from poke_bot.remote_sim_jobs import remote_promotion_job
    from poke_bot.worker_pool import WorkerPool

    n_games = int(n_games)
    if n_games < 4 or n_games % 2:
        raise ValueError("--promotion-games must be an even integer >= 4")
    if not decks:
        raise ValueError("promotion requires at least one deck")
    cfg = PromotionGateConfig(
        min_games=n_games,
        min_complete_pairs=n_games // 2,
        threshold=float(threshold),
        confidence=float(confidence),
        bootstrap_resamples=int(bootstrap_resamples),
    )
    cfg.validate()
    jobs: list[dict[str, Any]] = []
    for game_i in range(n_games):
        pair_i = game_i // 2
        _arch, deck = decks[pair_i % len(decks)]
        jobs.append(
            {
                "candidate_checkpoint": candidate.path,
                "parent_checkpoint": incumbent.path,
                "candidate_seat": game_i % 2,
                "cluster_id": pair_i,
                "seed": int(seed + game_i),
                "deck": list(deck),
                "device": "cpu",
                "agent_mode": "policy",
                "mcts_sims": 0,
                "mcts_move_time": 0.0,
                "candidate_digest": candidate.digest,
                "parent_digest": incumbent.digest,
                "model_generation": int(model_generation),
                "model_max_context": int(pure_rl_model_config().max_context),
                "timeout_s": int(game_timeout_s),
            }
        )
    with WorkerPool(num_workers=max(1, min(int(n_workers), n_games))) as pool:
        rows = list(pool.imap_unordered(remote_promotion_job, jobs))
    report = evaluate_candidate_gate(rows, cfg)
    report.update(
        {
            "candidate": candidate.as_dict(),
            "incumbent": incumbent.as_dict(),
            "deck_schedule": "paired_same_deck_both_candidate_seats",
            "search": False,
        }
    )
    return report, rows


def run_full_loop(args: argparse.Namespace) -> int:
    """Real CABT collect → AWR → held-out loop with optional remote farms."""
    import torch
    from poke_bot.baselines_runtime import (
        baseline_content_digest,
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.pure_rl.strong_public_gate import (
        build_active_gate_result,
        load_active_gate_contract,
        verify_roster_content,
    )
    from poke_bot.promotion import CheckpointIdentity
    from poke_bot.remote_jobs import RemoteWorkerFarm

    if int(args.expert_rehearsal_every) < 0:
        raise ValueError("--expert-rehearsal-every cannot be negative")
    if int(args.expert_rehearsal_force_before) < -1:
        raise ValueError("--expert-rehearsal-force-before must be -1 or nonnegative")
    if (
        int(args.expert_rehearsal_every) > 0
        or int(args.expert_rehearsal_force_before) >= 0
    ) and args.expert_manifest is None:
        raise ValueError("--expert-manifest is required when expert rehearsal is enabled")
    if int(args.expert_rehearsal_epochs) <= 0:
        raise ValueError("--expert-rehearsal-epochs must be positive")
    if float(args.expert_rehearsal_lr) <= 0.0:
        raise ValueError("--expert-rehearsal-lr must be positive")
    if int(args.expert_rehearsal_batch_size) <= 0:
        raise ValueError("--expert-rehearsal-batch-size must be positive")
    if int(args.train_games_per_batch) <= 0:
        raise ValueError("--train-games-per-batch must be positive")
    if int(args.train_max_decisions_per_batch) <= 0:
        raise ValueError("--train-max-decisions-per-batch must be positive")
    _scheduled_train_decision_cap(
        0,
        steady_cap=int(args.train_max_decisions_per_batch),
        warmup_cap=int(args.train_warmup_max_decisions_per_batch),
        warmup_iterations=int(args.train_warmup_iterations),
    )
    if not 0.0 <= float(args.continuous_learner_min_wr) <= 1.0:
        raise ValueError("--continuous-learner-min-wr must be in [0, 1]")
    if int(args.artifact_history_iterations) <= 0:
        raise ValueError("--artifact-history-iterations must be positive")
    if float(args.min_free_disk_gb) < 0.0:
        raise ValueError("--min-free-disk-gb cannot be negative")
    active_gate_contract: Optional[dict[str, Any]] = None
    active_gate_contract_identity: Optional[dict[str, Any]] = None
    active_gate: Optional[dict[str, Any]] = None
    require_active_contract = bool(
        args.mode == "specialist"
        or str(os.environ.get("POKEBOT_REQUIRE_ACTIVE_GATE_CONTRACT", "0")).lower()
        in ("1", "true", "yes", "on")
    )
    if args.active_gate_contract is None:
        if require_active_contract:
            raise RuntimeError(
                "production specialist launch requires --active-gate-contract"
            )
        args.heldout_games = int(args.heldout_games or 200)
    else:
        active_gate_path = Path(args.active_gate_contract).expanduser().resolve()
        active_gate_contract = load_active_gate_contract(active_gate_path)
        active_gate_contract_identity = _path_content_identity(active_gate_path)
        active_gate = dict(active_gate_contract["next_gate"])
        contract_games = int(active_gate["evaluation"]["games_total"])
        if args.heldout_games is not None and int(args.heldout_games) != contract_games:
            raise RuntimeError(
                "--heldout-games disagrees with active gate contract: "
                f"cli={args.heldout_games} contract={contract_games}"
            )
        args.heldout_games = contract_games
        args.gate_wr = float(
            active_gate["pass_criteria"]["skill_weighted_win_rate"]
        )
        args.heldout_per_opponent_floor = float(
            active_gate["pass_criteria"].get("individual_opponent_floor", 0.0)
        )
        print(
            "[pure_rl] ACTIVE_GATE_ASSERT "
            f"id={active_gate['id']} opponents={len(active_gate['roster'])} "
            f"games={contract_games} per_opponent="
            f"{active_gate['evaluation']['games_per_opponent']} "
            "stage=heldout:strong_public_gate",
            flush=True,
        )
    if int(args.heldout_games) <= 0:
        raise ValueError("heldout game count must be positive")

    hw = full_hardware_profile()
    if args.allow_single_gpu:
        hw = replace(hw, allow_single_gpu=True)
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw.validate_or_raise(visible_gpu_count=visible)

    run_dir = _run_dir(args.run_name)
    loop_state = _load_loop_state(run_dir)
    resumed = loop_state is not None
    immutable_manifest: Optional[dict[str, Any]] = None
    if loop_state is not None:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume ledger exists but immutable manifest is missing")
        immutable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ckpt, start_iteration = _validate_resume_state(
            args=args, run_dir=run_dir, state=loop_state
        )
        ckpt = _ensure_pure_rl_checkpoint(ckpt, args.seed, smoke=False)
        print(
            f"[pure_rl] RESUME next_iteration={start_iteration} "
            f"champion={ckpt}",
            flush=True,
        )
    else:
        if args.start_iteration not in (None, 0):
            raise RuntimeError(
                "a new lineage must start at iteration 0; use its loop ledger "
                "to resume a later iteration"
            )
        if (run_dir / "manifest.json").exists() or any(
            any((run_dir / part).glob("iter_*"))
            for part in ("shards", "checkpoints", "metrics", "eval")
        ):
            raise RuntimeError(
                f"run directory {run_dir} has artifacts but no loop_state.json; "
                "refusing destructive/inferred resume — use a new run name"
            )
        seed_path = (
            Path(args.base_checkpoint).expanduser().resolve()
            if args.base_checkpoint is not None
            else (run_dir / "checkpoints" / "seed.pt")
        )
        ckpt = _ensure_pure_rl_checkpoint(seed_path, args.seed, smoke=False)
        start_iteration = 0

    os.environ["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    os.environ.setdefault(
        "POKEBOT_PRIMARY_ARCHETYPE",
        str(args.specialist_archetype or "core"),
    )
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    identity = CheckpointIdentity.from_path(ckpt)
    endpoints = _resolve_remote_endpoints(args)

    ensure_baselines_installed()
    manifest_baselines = load_manifest()
    loadable, _failed_baselines = filter_loadable_baselines(manifest_baselines)
    by_id = {s.id: s for s in loadable}
    official_specs = [by_id[i] for i in OFFICIAL_BASELINE_IDS if i in by_id]
    if len(official_specs) < len(OFFICIAL_BASELINE_IDS):
        missing = [i for i in OFFICIAL_BASELINE_IDS if i not in by_id]
        raise RuntimeError(
            f"formal held-out gate is unavailable; missing official baselines: {missing}"
        )
    if active_gate is not None:
        active_gate_ids = tuple(
            str(row["opponent_id"]) for row in active_gate["roster"]
        )
        missing = [opponent_id for opponent_id in active_gate_ids if opponent_id not in by_id]
        if missing:
            raise RuntimeError(
                f"active gate packages are unavailable: {missing}"
            )
        heldout_specs = [by_id[opponent_id] for opponent_id in active_gate_ids]
        installed_gate_digests = {
            spec.id: baseline_content_digest(spec.path)
            for spec in heldout_specs
        }
        verify_roster_content(
            active_gate,
            installed_gate_digests,
        )
        active_gate_content_digests = set(installed_gate_digests.values())
    else:
        active_gate_ids = tuple(OFFICIAL_BASELINE_IDS)
        heldout_specs = list(official_specs)
        active_gate_content_digests = set()
    # The active strong-public roster is formal held-out evidence, not part of
    # the public training mixture.  Exclude both the legacy research controls
    # and every contract-selected gate opponent so the gate remains genuinely
    # disjoint from the candidate's training data.
    heldout_ids = set(OFFICIAL_BASELINE_IDS) | set(active_gate_ids)
    collect_candidates = [spec for spec in loadable if spec.id not in heldout_ids]
    collect_content_digests = (
        {
            spec.id: baseline_content_digest(spec.path)
            for spec in collect_candidates
        }
        if active_gate_content_digests
        else {}
    )
    excluded_gate_digest_aliases = sorted(
        spec.id
        for spec in collect_candidates
        if collect_content_digests.get(spec.id) in active_gate_content_digests
    )
    collect_specs = [
        spec
        for spec in collect_candidates
        if collect_content_digests.get(spec.id) not in active_gate_content_digests
    ]
    if active_gate is not None:
        print(
            "[pure_rl] ACTIVE_GATE_TRAINING_DISJOINT "
            f"gate_ids={len(active_gate_ids)} "
            f"gate_digests={len(active_gate_content_digests)} "
            f"excluded_digest_aliases={excluded_gate_digest_aliases} "
            f"collect_ids={len(collect_specs)}",
            flush=True,
        )
    if not collect_specs:
        raise RuntimeError(
            "no training opponents remain after excluding the formal held-out set"
        )
    ladder_mix = None
    ladder_representatives = None
    ladder_contract: Optional[dict[str, Any]] = None
    if args.mode == "core":
        (
            decks,
            ladder_mix,
            ladder_representatives,
            ladder_contract,
        ) = _core_ladder_decks()
    else:
        decks = _our_decks(args.mode, args.specialist_archetype)
    measurement_decks = _select_measurement_decks(decks, args.measurement_decks)
    deck_names = [n for n, _ in decks]
    measurement_deck_names = [n for n, _ in measurement_decks]
    print(
        f"[pure_rl] our_decks n_decks={len(decks)} sample={deck_names[:16]}"
        + (
            f" ladder_mix={ladder_mix.artifact_sha256}"
            if ladder_mix is not None
            else ""
        ),
        flush=True,
    )
    print(
        "[pure_rl] measurement_decks "
        f"n_decks={len(measurement_decks)} ids={measurement_deck_names}",
        flush=True,
    )

    from poke_bot.remote_sim_jobs import (
        remote_play_job as worker_play,
        remote_self_play_job as worker_self_play,
        remote_self_play_multi_job as worker_self_play_multi,
    )
    from poke_bot.pure_rl.multi_env_self_play import (
        process_worker_count,
        pure_rl_leaf_coalesce_ms,
        resolve_multi_env_per_worker,
    )

    multi_env_n = resolve_multi_env_per_worker(args.multi_env_per_worker)
    proc_workers = process_worker_count(hw.sim_workers, multi_env_n)
    if args.leaf_coalesce_ms is not None:
        coalesce_ms = float(args.leaf_coalesce_ms)
    else:
        coalesce_ms = pure_rl_leaf_coalesce_ms(default=0.0)

    from poke_bot.live_pool import LIVE_POOL_PLAN_PATH, live_pool_enabled
    from poke_bot.pure_rl.live_pool_apply import apply_live_pool_plan

    live_pool_on = live_pool_enabled()

    checkpoint_profile = _checkpoint_contract(ckpt, smoke=False)
    initial_replay_shards = [
        Path(path).expanduser().resolve() for path in args.initial_replay_shard
    ]
    for shard in initial_replay_shards:
        if not shard.is_file():
            raise FileNotFoundError(f"initial replay shard is missing: {shard}")
    initial_replay_contract = [
        _path_content_identity(path) for path in initial_replay_shards
    ]
    if resumed:
        assert loop_state is not None
        initial_learner_identity = _verified_checkpoint_identity(
            loop_state.get("learner") or loop_state.get("champion") or {}
        )
        assert immutable_manifest is not None
        stored_initial_learner = (
            (immutable_manifest.get("design_contract") or {})
            .get("learner", {})
            .get("initial_checkpoint")
        )
        design_initial_learner_identity = _verified_checkpoint_identity(
            stored_initial_learner
        )
        if args.initial_learner_checkpoint is not None:
            requested_learner = CheckpointIdentity.from_path(
                Path(args.initial_learner_checkpoint).expanduser().resolve()
            )
            if requested_learner.digest != design_initial_learner_identity.digest:
                raise RuntimeError(
                    "--initial-learner-checkpoint conflicts with original lineage: "
                    f"requested={requested_learner.digest} "
                    f"original={design_initial_learner_identity.digest}"
                )
    else:
        initial_learner_identity = (
            CheckpointIdentity.from_path(
                Path(args.initial_learner_checkpoint).expanduser().resolve()
            )
            if args.initial_learner_checkpoint is not None
            else identity
        )
        design_initial_learner_identity = initial_learner_identity
        learner_profile = _checkpoint_contract(
            initial_learner_identity.path, smoke=False
        )
        for field in (
            "model_profile",
            "decision_context",
            "max_context",
            "feature_schema",
        ):
            if learner_profile[field] != checkpoint_profile[field]:
                raise RuntimeError(
                    "initial learner is incompatible with rollout champion: "
                    f"field={field}"
                )
    initial_heldout_evidence_contract: Optional[dict[str, Any]] = None
    initial_heldout_evidence: Optional[dict[str, Any]] = None
    if args.initial_heldout_evidence is not None:
        initial_heldout_evidence_path = (
            Path(args.initial_heldout_evidence).expanduser().resolve()
        )
        initial_heldout_evidence_contract = _path_content_identity(
            initial_heldout_evidence_path
        )
        # The accepted original-four report remains immutable lineage history,
        # but it is not evidence for the active contract and must never seed
        # the active-gate champion selector.
        if active_gate is None:
            initial_heldout_evidence = _load_initial_heldout_evidence(
                initial_heldout_evidence_path,
                checkpoint=design_initial_learner_identity,
                heldout_games=int(args.heldout_games),
            )
    source_snapshot = _source_snapshot()
    if resumed:
        assert loop_state is not None and immutable_manifest is not None
        lineage_base = _verified_checkpoint_identity(
            loop_state.get("lineage_base") or {}
        ).as_dict()
    else:
        lineage_base = identity.as_dict()
    design_contract = _design_contract(
        args=args,
        checkpoint_profile=checkpoint_profile,
        lineage_base=lineage_base,
        initial_learner=design_initial_learner_identity.as_dict(),
        initial_heldout_evidence=initial_heldout_evidence_contract,
        initial_replay_shards=initial_replay_contract,
        source=source_snapshot,
        decks=decks,
        measurement_decks=measurement_decks,
        ladder_contract=ladder_contract,
        endpoints=endpoints,
        collect_specs=collect_specs,
        official_specs=official_specs,
        heldout_specs=heldout_specs,
        multi_env_per_worker=multi_env_n,
        leaf_coalesce_ms=coalesce_ms,
    )
    design_fingerprint = _design_fingerprint(design_contract)

    if not resumed:
        loop_state = {
            "version": LOOP_STATE_VERSION,
            "run_name": args.run_name,
            "mode": args.mode,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "next_iteration": 0,
            "last_completed_iteration": -1,
            "champion": identity.as_dict(),
            "heldout_champion": (
                initial_learner_identity.as_dict()
                if initial_heldout_evidence is not None
                else identity.as_dict()
            ),
            "heldout_champion_evidence": initial_heldout_evidence,
            "learner": initial_learner_identity.as_dict(),
            "lineage_base": lineage_base,
            "design_fingerprint": design_fingerprint,
            "opponent_pool": [identity.as_dict()],
            "raw_advantage_history": [],
            "policy_prev_agreement_history": [],
            "history": [],
            "live_pool_last_seq": 0,
        }
        _atomic_json(run_dir / "loop_state.json", loop_state)
        manifest = {
                "run_name": args.run_name,
                "mode": args.mode,
                "specialist_archetype": args.specialist_archetype,
                "smoke": False,
                "hardware": hw.as_dict(),
                "stage": stage_to_dict(stage),
                "base_checkpoint": str(ckpt),
                "checkpoint_digest": identity.digest,
                "base_checkpoint_contract": checkpoint_profile,
                "initial_learner_checkpoint": initial_learner_identity.as_dict(),
                "initial_heldout_evidence": initial_heldout_evidence_contract,
                "initial_replay_shards": initial_replay_contract,
                "model_profile": checkpoint_profile["model_profile"],
                "trainable_parameters": int(
                    checkpoint_profile["trainable_parameters"]
                ),
                "decision_context": checkpoint_profile["decision_context"],
                "max_context": checkpoint_profile["max_context"],
                "feature_schema": checkpoint_profile["feature_schema"],
                "design_fingerprint": design_fingerprint,
                "design_contract": design_contract,
                "param_fail_max": int(config.PURE_RL.param_fail_max),
                "gate_wr": float(args.gate_wr),
                "heldout_games": int(args.heldout_games),
                "heldout_per_opponent_floor": float(
                    args.heldout_per_opponent_floor
                ),
                "promotion_gate": {
                    "games": int(args.promotion_games),
                    "workers": int(args.promotion_workers),
                    "threshold": float(args.promotion_threshold),
                    "confidence": float(args.promotion_confidence),
                    "bootstrap_resamples": int(
                        args.promotion_bootstrap_resamples
                    ),
                },
                "training_design": {
                    "spinning_up": "https://spinningup.openai.com/en/latest/spinningup/spinningup.html",
                    "algorithm": "AWR_actor_critic",
                    "not_alphazero": True,
                    "gamma": 1.0,
                    "return_type": "terminal_monte_carlo",
                    "awr_baseline": "frozen_precomputed_per_iteration",
                    "normalize_advantages": bool(config.PURE_RL.normalize_advantages),
                    "entropy_bonus": float(config.PURE_RL.entropy_bonus),
                    "bootstrap_mix": float(config.PURE_RL.bootstrap_mix),
                    "replay_window_shards": int(config.PURE_RL.replay_window_shards),
                    "self_play_frac": float(config.PURE_RL.self_play_frac),
                    "collect_temperature": float(args.collect_temperature),
                    "collect_temperature_final": float(
                        config.PURE_RL.collect_temperature_final
                    ),
                    "train_epochs": int(args.train_epochs),
                },
                "leaf_devices": hw.leaf_cuda_devices(),
                "remote_worker_endpoints": endpoints,
                "collect_opponents": [s.id for s in collect_specs],
                "official_target_training_opponents": (
                    [s.id for s in official_specs]
                    if float(args.official_collect_frac) > 0.0
                    else []
                ),
                "active_gate_id": (
                    str(active_gate["id"]) if active_gate is not None else None
                ),
                "active_gate_contract": active_gate_contract_identity,
                "heldout_opponents": list(active_gate_ids),
                "our_decks": deck_names,
                "n_decks": len(decks),
                "ladder_deck_mix": ladder_contract,
                "multi_env_per_worker": multi_env_n,
                "proc_workers_self_play": proc_workers,
                "leaf_coalesce_ms": coalesce_ms,
                "live_pool": live_pool_on,
                "live_pool_plan": str(LIVE_POOL_PLAN_PATH),
                "min_usable_game_frac": float(args.min_usable_game_frac),
                "source": source_snapshot,
                "selected_environment": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith(("PURE_RL_", "POKEBOT_"))
                    or key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
                },
                "note": (
                    "Single small AWR trainee (~1-3M); remotes are whole-game "
                    "collect farms merging into the same shards. Kaggle reports "
                    "motivate experiments but do not prescribe these hyperparameters."
                ),
        }
        _write_json_exclusive(run_dir / "manifest.json", manifest)
    else:
        assert loop_state is not None and immutable_manifest is not None
        # Quarantine any uncommitted partial N+1 attempt before evaluating a
        # clean-boundary gate migration. Otherwise a stale partial shard can
        # block the very launch that is supposed to replace its bad contract.
        recovery = _recover_interrupted_iteration(run_dir, loop_state)
        if recovery is not None:
            print(
                f"[pure_rl] recovered interrupted iteration into {recovery}",
                flush=True,
            )
        effective_before_migration, _effective_digest, _migration_receipts = (
            _load_design_migration_chain(run_dir, immutable_manifest)
        )
        recovered_collection = _ensure_recoverable_completed_collection(
            run_dir,
            loop_state,
            effective_before_migration,
        )
        if recovered_collection is not None:
            print(
                "[pure_rl] completed collection recovery proof ready "
                f"iter={recovered_collection['iteration']} "
                f"receipt={recovered_collection['receipt_path']}",
                flush=True,
            )
        design_fingerprint = _validate_or_migrate_design_fingerprint(
            run_dir=run_dir,
            state=loop_state,
            manifest=immutable_manifest,
            current=design_contract,
            allow_clean_boundary_migration=bool(
                args.allow_clean_boundary_design_migration
            ),
            migration_reason=(
                os.environ.get("PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE")
                or args.boundary_design_migration_reason
            ),
        )

    terminal_marker = _ensure_terminal_gate_marker(run_dir, loop_state)
    if terminal_marker is not None:
        print(
            f"[pure_rl] committed terminal gate already complete: {terminal_marker}",
            flush=True,
        )
        return 0

    assert loop_state is not None
    learner_identity = _verified_checkpoint_identity(
        loop_state.get("learner") or loop_state.get("champion") or {}
    )
    heldout_champion_identity = _verified_checkpoint_identity(
        loop_state.get("heldout_champion") or loop_state.get("champion") or {}
    )
    raw_heldout_champion_evidence = dict(
        loop_state.get("heldout_champion_evidence") or {}
    )
    heldout_champion_evidence = _reconciled_heldout_champion_evidence(
        loop_state
    )
    if heldout_champion_evidence != raw_heldout_champion_evidence:
        loop_state["heldout_champion_evidence"] = heldout_champion_evidence
        print(
            "[pure_rl] reconciled legacy heldout champion evidence from "
            "committed per-opponent gate",
            flush=True,
        )
    if heldout_champion_evidence and str(
        heldout_champion_evidence.get("checkpoint_digest") or ""
    ) != str(heldout_champion_identity.digest):
        raise RuntimeError(
            "heldout champion evidence digest does not match its checkpoint"
        )
    live_pool_last_seq = int(loop_state.get("live_pool_last_seq", 0))
    opponent_pool_entries = list(loop_state.get("opponent_pool") or [])
    opponent_pool = [
        _verified_checkpoint_identity(entry) for entry in opponent_pool_entries
    ]
    if not any(entry.digest == identity.digest for entry in opponent_pool):
        opponent_pool.append(identity)
    adv_hist: list[float] = [
        float(x) for x in (loop_state.get("raw_advantage_history") or [])
    ]
    agr_hist: list[float] = [
        float(x)
        for x in (loop_state.get("policy_prev_agreement_history") or [])
    ]

    print(
        f"[pure_rl] full hardware workers={hw.sim_workers} "
        f"self_play_procs={proc_workers} multi_env={multi_env_n} "
        f"leaf_coalesce_ms={coalesce_ms} "
        f"leaves_gpu0={hw.leaf_gpu0_replicas} leaves_gpu1={hw.leaf_gpu1_replicas} "
        f"train_cuda={hw.train_cuda_device} remotes={endpoints or 'none'} "
        f"live_pool={'on' if live_pool_on else 'off'} "
        f"champion={identity.digest[:19]}… "
        f"heldout_champion={heldout_champion_identity.digest[:19]}… "
        f"learner={learner_identity.digest[:19]}…",
        flush=True,
    )

    leaf = _LeafFarm()
    remote_farm: Optional[RemoteWorkerFarm] = None
    train_dev = torch.device(f"cuda:{hw.train_cuda_device}")
    expert_cache = ResidentExpertCorpusCache()

    try:
        if args.leaf_eval == "gpu-server" and visible >= 1:
            leaf_devices = hw.leaf_cuda_devices()
            if visible < 2:
                leaf_devices = [0] * max(1, hw.leaf_gpu0_replicas)
            # Resp queues cover max(process counts) used by self-play or baseline pools.
            leaf_slots = max(proc_workers, hw.sim_workers)
            # max_batch=None → each server uses leaf_batch_for_device (256 on
            # 3080 Ti / 512 on Blackwell). A shared 1024 blows 12GB GPU0.
            leaf.start(
                ckpt=learner_identity.path,
                digest=learner_identity.digest,
                leaf_devices=leaf_devices,
                n_workers=leaf_slots,
                max_batch=None,
                coalesce_ms=coalesce_ms,
            )
        else:
            print("[pure_rl] leaf-eval=cpu (local workers load model)", flush=True)

        if endpoints:
            remote_job_buffer_s = float(
                os.environ.get("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", "600") or "600"
            )
            remote_job_timeout = max(
                float(getattr(config.SEARCH, "remote_request_timeout_s", 120.0)),
                float(args.game_timeout_s) + remote_job_buffer_s,
            )
            remote_farm = RemoteWorkerFarm(endpoints, timeout_s=remote_job_timeout)
            require_all = str(
                os.environ.get("POKEBOT_REMOTE_REQUIRE_ALL", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
            no_local_fallback = str(
                os.environ.get("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
            try:
                infos = remote_farm.connect(require_all=require_all)
            except Exception as exc:
                print(
                    f"[pure_rl] ERROR: remote connect failed ({exc}); "
                    "fail-closed for production remotes requirement",
                    file=sys.stderr,
                    flush=True,
                )
                (run_dir / "REMOTE_CONNECT_FAILED").write_text(
                    json.dumps({"error": str(exc), "endpoints": endpoints}, indent=2),
                    encoding="utf-8",
                )
                try:
                    remote_farm.close()
                except Exception:
                    pass
                if require_all or no_local_fallback:
                    raise RuntimeError(
                        "required remote farm did not connect; refusing local-only "
                        "production collection"
                    ) from exc
                remote_farm = None
                infos = []
            for info in infos:
                print(
                    f"[pure_rl] remote={info.endpoint} host={info.hostname} "
                    f"gpu={info.gpu_name!r} workers={info.workers}",
                    flush=True,
                )
            if remote_farm is not None:
                print(
                    f"[pure_rl] remote additive capacity={remote_farm.total_workers}",
                    flush=True,
                )
                # Boot boundary: same hard gate as between-iter (no soft WARN).
                # Local leaves just started on this digest — skip redundant reload.
                _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=remote_farm,
                    ckpt=learner_identity.path,
                    digest=learner_identity.digest,
                    version=int(start_iteration) * 10,
                    required_endpoints=endpoints,
                    reload_local=False,
                )

        if args.heldout_remotes:
            if remote_farm is None:
                raise RuntimeError(
                    "--heldout-remotes requires a connected remote farm; "
                    "refusing to run a partially local formal gate"
                )
            heldout_remote_audit = _remote_heldout_capability_audit(
                remote_farm,
                required_endpoints=endpoints,
            )
            _atomic_json(
                run_dir / "heldout_remote_capability.json",
                heldout_remote_audit,
            )
            if not heldout_remote_audit["passed"]:
                raise RuntimeError(
                    "formal heldout remote capability audit failed: "
                    + json.dumps(heldout_remote_audit, sort_keys=True)
                )
            print(
                "[pure_rl] formal heldout remotes VERIFIED "
                f"endpoints={len(heldout_remote_audit['endpoints'])}",
                flush=True,
            )

        def _retain_completed_artifacts(
            state: dict[str, Any], completed_iteration: int
        ) -> dict[str, Any]:
            report = apply_artifact_retention(
                run_dir,
                state,
                completed_iteration=int(completed_iteration),
                replay_window_shards=int(config.PURE_RL.replay_window_shards),
                history_iterations=int(args.artifact_history_iterations),
            )
            _atomic_json(run_dir / "artifact_receipts" / "latest.json", report)
            if report.get("retired_shards") or report.get("retired_checkpoints"):
                print(
                    "[pure_rl] artifact retention "
                    f"shards={report.get('retired_shards')} "
                    f"checkpoints={report.get('retired_checkpoints')} "
                    f"reclaimed_gib={float(report.get('reclaimed_bytes', 0)) / (1024 ** 3):.2f}",
                    flush=True,
                )
            return report

        if int(loop_state.get("last_completed_iteration", -1)) >= 0:
            _retain_completed_artifacts(
                loop_state,
                int(loop_state["last_completed_iteration"]),
            )

        pending_collect: Optional[dict[str, Any]] = None

        def _rebuild_leaves_if_needed(champion: Path, dig: str) -> None:
            """Restart leaf farm after live-pool topology change (iter boundary)."""
            if args.leaf_eval != "gpu-server" or visible < 1:
                return
            leaf_devices = hw.leaf_cuda_devices()
            if visible < 2:
                leaf_devices = [0] * max(1, hw.leaf_gpu0_replicas or hw.leaf_replicas_total)
            leaf_slots = max(proc_workers, hw.sim_workers)
            print(
                f"[pure_rl] live_pool rebuild leaves "
                f"gpu0={hw.leaf_gpu0_replicas} gpu1={hw.leaf_gpu1_replicas} "
                f"slots={leaf_slots}",
                flush=True,
            )
            leaf.stop()
            leaf.start(
                ckpt=champion,
                digest=dig,
                leaf_devices=leaf_devices,
                n_workers=leaf_slots,
                max_batch=None,
                coalesce_ms=coalesce_ms,
            )

        def _maybe_apply_live_pool(
            champion: Path, dig: str, *, allow_leaf_rebuild: bool = True
        ) -> None:
            nonlocal hw, proc_workers, live_pool_last_seq
            new_hw, new_proc, new_seq, plan, leaf_changed = apply_live_pool_plan(
                hw=hw,
                last_seq=live_pool_last_seq,
                multi_env_per_worker=multi_env_n,
                visible_gpu_count=visible,
            )
            if plan is None:
                return
            # Response queues are 1:1 with WorkerPool slots. Growing workers
            # without growing resp_qs → ValueError queues=32 workers=96 and a
            # dead train (2026-07-16). leaf_changed only tracks GPU topology;
            # also detect slot/queue undersize.
            n_qs = len(leaf.resp_qs) if leaf.resp_qs else 0
            slots_need = max(int(new_hw.sim_workers), int(new_proc))
            slots_changed = bool(n_qs > 0 and slots_need > n_qs)
            need_rebuild = bool(leaf_changed or slots_changed)
            # Defer leaf *topology* rebuild at boot (CUDA teardown race), but
            # never adopt a worker count above current queue capacity.
            do_leaf = bool(
                need_rebuild and allow_leaf_rebuild and leaf.remote_channel
            )
            deferred = bool(need_rebuild and not do_leaf)
            print(
                f"[pure_rl] live_pool_plan seq={plan.seq} apply "
                f"workers={hw.sim_workers}->{new_hw.sim_workers} "
                f"leaves={hw.leaf_replicas_total}->{new_hw.leaf_replicas_total} "
                f"procs={proc_workers}->{new_proc}"
                + (f" reason={plan.reason}" if plan.reason else "")
                + (
                    f" (leaf rebuild deferred; clamp workers to queues={n_qs})"
                    if deferred and n_qs > 0
                    else (
                        " (leaf rebuild deferred)"
                        if deferred
                        else ""
                    )
                ),
                flush=True,
            )
            if deferred:
                # Keep current leaf topology AND clamp workers to queue slots.
                clamp_w = min(int(new_hw.sim_workers), n_qs) if n_qs > 0 else int(
                    new_hw.sim_workers
                )
                clamp_proc = min(int(new_proc), n_qs) if n_qs > 0 else int(new_proc)
                new_hw = replace(
                    new_hw,
                    leaf_gpu0_replicas=hw.leaf_gpu0_replicas,
                    leaf_gpu1_replicas=hw.leaf_gpu1_replicas,
                    sim_workers=max(1, clamp_w),
                    games_in_flight=max(1, clamp_w),
                )
                new_proc = max(1, clamp_proc)
            hw = new_hw
            proc_workers = new_proc
            # A deferred topology plan is not consumed. Re-read it at the next
            # safe iteration boundary, where the leaf farm can actually rebuild.
            if not deferred:
                live_pool_last_seq = new_seq
            if do_leaf:
                _rebuild_leaves_if_needed(champion, dig)

        def _kick_collect(it: int, champion: Path, dig: str) -> dict[str, Any]:
            import shutil

            free_bytes = int(shutil.disk_usage(run_dir).free)
            minimum_free = int(float(args.min_free_disk_gb) * (1024 ** 3))
            if free_bytes < minimum_free:
                raise RuntimeError(
                    "free-disk guard refused a new replay shard: "
                    f"free={free_bytes / (1024 ** 3):.1f}GiB "
                    f"minimum={minimum_free / (1024 ** 3):.1f}GiB"
                )
            temp = _collect_temperature(args, it)
            pool_n = max(1, int(getattr(config.PURE_RL, "opponent_pool_size", 4)))
            recent = [
                _verified_checkpoint_identity(entry)
                for entry in opponent_pool[-pool_n:]
            ]
            official_target_weights: Optional[dict[str, float]] = None
            if bool(args.official_adaptive_targeting):
                exact_rates = _latest_official_heldout_win_rates(loop_state)
                official_target_weights = _adaptive_official_target_weights(
                    tuple(str(spec.id) for spec in official_specs),
                    exact_rates,
                    target_win_rate=float(args.heldout_per_opponent_floor),
                    minimum_share=float(args.official_adaptive_min_share),
                    gap_power=float(args.official_adaptive_gap_power),
                )
                print(
                    "[pure_rl] adaptive official targeting "
                    f"source={'latest_exact_heldout' if exact_rates else 'uniform_fallback'} "
                    f"rates={json.dumps(exact_rates, sort_keys=True)} "
                    f"weights={json.dumps(official_target_weights, sort_keys=True)}",
                    flush=True,
                )
            self_jobs, base_jobs = _build_collect_jobs(
                n_games=args.games_per_iter,
                ckpt=champion,
                digest=dig,
                model_generation=it + 1,
                decks=decks,
                specs=collect_specs,
                seed=args.seed + it * 100_000,
                game_timeout_s=args.game_timeout_s,
                mode=args.mode,
                collect_temperature=temp,
                max_context=pure_rl_model_config().max_context,
                opponent_pool=recent,
                self_play_frac=float(getattr(config.PURE_RL, "self_play_frac", 0.85)),
                ladder_mix=ladder_mix,
                iteration=int(it),
                priority_specs=(
                    official_specs
                    if float(args.official_collect_frac) > 0.0
                    else None
                ),
                priority_frac=float(args.official_collect_frac),
                priority_weights=official_target_weights,
                official_exploit_opponents=tuple(
                    args.official_exploit_opponents
                ),
                official_exploit_frac=float(args.official_exploit_frac),
                official_exploit_temperature=float(
                    args.official_exploit_temperature
                ),
            )
            shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
            if shard.is_file():
                raise RuntimeError(
                    f"refusing to overwrite immutable collect shard: {shard}"
                )
            collect_started = time.time()
            _atomic_json(
                run_dir / "iteration_runtime.json",
                {
                    "schema": "poke_bot.iteration_runtime/v1",
                    "iteration": int(it),
                    "phase": "collect",
                    "started_at": collect_started,
                    "checkpoint_digest": str(dig),
                    "updated_at": collect_started,
                },
            )
            from collections import Counter

            public_groups = Counter(
                str(
                    (job.get("target_provenance") or {}).get(
                        "opponent_training_group", "diverse_public"
                    )
                )
                for job in base_jobs
            )
            print(
                f"[pure_rl] collect iter={it} self_play={len(self_jobs)} "
                f"public_mix={len(base_jobs)} "
                f"official_target={public_groups.get('official_target', 0)} "
                f"diverse_public={public_groups.get('diverse_public', 0)} "
                f"n_decks={len(decks)} sample={deck_names[:12]} "
                f"local_workers={hw.sim_workers} "
                f"self_play_procs={proc_workers} multi_env={multi_env_n} "
                f"remote_workers="
                f"{remote_farm.total_workers if remote_farm else 0}",
                flush=True,
            )
            writer, rows, stats = _collect_wave(
                self_play_jobs=self_jobs,
                baseline_jobs=base_jobs,
                shard_path=shard,
                n_workers=hw.sim_workers,
                leaf_channel=leaf.remote_channel,
                remote_farm=remote_farm,
                worker_play=worker_play,
                worker_self_play=worker_self_play,
                multi_env_per_worker=multi_env_n,
                worker_self_play_multi=worker_self_play_multi,
                iteration=int(it),
            )
            collect_elapsed = max(time.time() - collect_started, 1e-6)
            requested = len(self_jobs) + len(base_jobs)
            retained_jobs = int(stats.get("with_record", 0))
            min_retained = math.ceil(requested * float(args.min_usable_game_frac))
            stats.update(
                {
                    "requested_games": requested,
                    "retained_source_games": retained_jobs,
                    "retained_trajectories": writer.n_games,
                    "usable_game_fraction": (
                        retained_jobs / requested if requested else 1.0
                    ),
                    "collect_elapsed_sec": collect_elapsed,
                    "claimed_games_per_sec": requested / collect_elapsed,
                    "valid_source_games_per_sec": retained_jobs / collect_elapsed,
                    "trajectory_games_per_sec": writer.n_games / collect_elapsed,
                }
            )
            print(
                f"[pure_rl] collect done iter={it} ok={stats.get('ok')} "
                f"leaf_remote={stats.get('leaf_remote')} "
                f"leaf_modes={stats.get('leaf_modes')} "
                f"multi_env_games={stats.get('multi_env_games')}",
                flush=True,
            )
            if retained_jobs < min_retained:
                failed = shard.with_name(
                    f"{shard.stem}.failed."
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    f"{shard.suffix}"
                )
                if shard.exists():
                    shard.replace(failed)
                _write_json_exclusive(
                    failed.with_suffix(failed.suffix + ".failure.json"),
                    {
                        "iteration": it,
                        "reason": "usable_game_fraction_below_threshold",
                        "threshold": float(args.min_usable_game_frac),
                        "stats": stats,
                        "quarantined_shard": str(failed),
                    },
                )
                raise RuntimeError(
                    f"collect iter={it} retained {retained_jobs}/{requested} "
                    f"source games (< {min_retained}); partial shard quarantined "
                    f"at {failed}"
                )
            stats["collect_completed_at"] = time.time()
            collection_receipt = _commit_completed_collection_receipt(
                run_dir=run_dir,
                state=loop_state,
                contract=design_contract,
                iteration=it,
                shard=shard,
                checkpoint=champion,
                checkpoint_digest=dig,
                stats=stats,
                started_at=collect_started,
                writer=writer,
            )
            return {
                "iteration": it,
                "shard": shard,
                "writer": writer,
                "stats": stats,
                "checkpoint": champion,
                "digest": dig,
                "started_at": collect_started,
                "receipt": collection_receipt["receipt_path"],
            }

        def _prepare_expert_rehearsal(
            it: int, parent: Any
        ) -> tuple[Any, dict[str, Any]]:
            """Recover or run the immutable ladder complement pass."""
            loss_weights = {
                "value": 1.0,
                "archetype": float(args.archetype_aux_loss_weight),
                "opponent_hand": float(args.opp_hand_loss_weight),
                "opponent_hidden_remainder": float(
                    args.opp_remainder_loss_weight
                ),
                "lethal_threat": float(args.lethal_threat_loss_weight),
                "prize_race": float(args.prize_race_loss_weight),
                "alakazam_guide": float(args.alakazam_guide_loss_weight),
            }
            corpus_split_seed = int(args.seed) + 5_000_000
            from poke_bot import checkpoint as checkpoint_mod

            trusted_parent = checkpoint_mod.assert_trusted_policy_checkpoint(
                parent.path
            )
            parent_profile = dict(trusted_parent.get("model_config") or {})
            rehearsal_context = (
                int(parent_profile.get("max_context") or 0)
                if trusted_parent.get("decision_context") == "history"
                else None
            )
            if rehearsal_context is not None and rehearsal_context <= 0:
                raise RuntimeError(
                    "temporal expert rehearsal requires a positive max_context"
                )
            exact_rehearsal_enabled = any(
                float(weight) > 0.0
                for weight in (
                    args.archetype_aux_loss_weight,
                    args.opp_hand_loss_weight,
                    args.opp_remainder_loss_weight,
                    args.lethal_threat_loss_weight,
                    args.prize_race_loss_weight,
                )
            )
            rehearsal_belief_card_vocab: Optional[int] = None
            if exact_rehearsal_enabled:
                # ``assert_trusted_policy_checkpoint`` deliberately returns
                # only validated metadata.  Read tensors from the already
                # trusted checkpoint instead of assuming the metadata view
                # contains ``model_state_dict``.
                parent_checkpoint = checkpoint_mod.load_checkpoint(
                    parent.path, map_location="cpu"
                )
                parent_state = dict(
                    parent_checkpoint.get("model_state_dict") or {}
                )
                rehearsal_belief_card_vocab = belief_card_vocab_from_state(
                    parent_state
                )
                missing_rehearsal_heads = sorted(
                    name
                    for name in (
                        "opp_hand_head",
                        "opp_remainder_head",
                        "lethal_threat_head",
                        "prize_race_head",
                    )
                    if f"{name}.weight" not in parent_state
                )
                if missing_rehearsal_heads:
                    print(
                        "[pure_rl] EXPERT_HEAD_WARM_START "
                        f"before_iter={it} missing={missing_rehearsal_heads} "
                        f"belief_card_vocab={rehearsal_belief_card_vocab} "
                        "optimizer=fresh",
                        flush=True,
                    )
            manifest_identity = resolve_expert_manifest(
                Path(args.expert_manifest),
                min_decisions=int(args.expert_min_decisions),
                require_protected=True,
                required_archetype=str(args.specialist_archetype or ""),
                required_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
                required_max_context=rehearsal_context,
                required_target_coverage=(
                    "temporal_action_rows",
                    "opponent_hand_rows",
                    "opponent_remainder_rows",
                    "opponent_private_prize_rows",
                    "lethal_threat_rows",
                    "prize_race_rows",
                ),
            )
            record = recover_rehearsal(
                run_dir,
                before_iteration=it,
                parent_digest=parent.digest,
                epochs=int(args.expert_rehearsal_epochs),
                learning_rate=float(args.expert_rehearsal_lr),
                manifest_identity=manifest_identity,
                loss_weights=loss_weights,
                corpus_split_seed=corpus_split_seed,
            )
            if record is None:
                print(
                    f"pure_rl train:expert iter={it}:   0%|"
                    "                                    | 0/1 "
                    "[00:00<?, ?expert pass/s, loading corpus]",
                    file=sys.stderr,
                    flush=True,
                )
                corpus = expert_cache.prepare(
                    manifest_identity,
                    device=train_dev,
                    seed=corpus_split_seed,
                    max_context=rehearsal_context,
                    belief_card_vocab=rehearsal_belief_card_vocab,
                )
                rehearsal_checkpoint, _receipt_path = rehearsal_paths(run_dir, it)
                print(
                    f"[pure_rl] expert rehearsal begin before_iter={it} "
                    f"decisions={manifest_identity.decisions} "
                    f"days={manifest_identity.dates[0]}..{manifest_identity.dates[-1]} "
                    f"parent={parent.digest[:19]}…",
                    flush=True,
                )
                rehearsal_result = supervised_rehearsal_step(
                    corpus,
                    base_ckpt=parent.path,
                    output_path=rehearsal_checkpoint,
                    parent_digest=parent.digest,
                    rehearsal_iteration=it,
                    manifest_identity=manifest_identity.as_dict(),
                    epochs=int(args.expert_rehearsal_epochs),
                    lr=float(args.expert_rehearsal_lr),
                    requested_batch_size=int(args.expert_rehearsal_batch_size),
                    seed=args.seed + 5_100_000 + it,
                    corpus_split_seed=corpus_split_seed,
                    device=train_dev,
                    aux_loss_weight=float(args.archetype_aux_loss_weight),
                    opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                    opp_remainder_loss_weight=float(
                        args.opp_remainder_loss_weight
                    ),
                    lethal_threat_loss_weight=float(
                        args.lethal_threat_loss_weight
                    ),
                    prize_race_loss_weight=float(args.prize_race_loss_weight),
                    alakazam_guide_loss_weight=float(
                        args.alakazam_guide_loss_weight
                    ),
                )
                record = commit_rehearsal_receipt(
                    run_dir,
                    before_iteration=it,
                    parent_digest=parent.digest,
                    manifest=manifest_identity,
                    epochs=int(args.expert_rehearsal_epochs),
                    learning_rate=float(args.expert_rehearsal_lr),
                    loss_weights=loss_weights,
                    corpus_split_seed=corpus_split_seed,
                    result=rehearsal_result,
                )
            prepared = _verified_checkpoint_identity(record["checkpoint_identity"])
            print(
                f"[pure_rl] expert rehearsal committed before_iter={it} "
                f"checkpoint={prepared.digest[:19]}… "
                f"reused={int(bool(record.get('reused')))}",
                flush=True,
            )
            return prepared, record

        # Start exactly at the ledger boundary. Collection is synchronous and
        # never overlaps a train/promotion/reload boundary.
        if start_iteration >= int(args.iterations):
            print(
                f"[pure_rl] ledger already complete next={start_iteration} "
                f"iterations={args.iterations}",
                flush=True,
            )
            return 0
        precollect_rehearsals: dict[int, dict[str, Any]] = {}
        if int(start_iteration) == 0 and bool(args.expert_rehearsal_before_first):
            learner_identity, first_record = _prepare_expert_rehearsal(
                0, learner_identity
            )
            precollect_rehearsals[0] = first_record
        _maybe_apply_live_pool(
            learner_identity.path,
            learner_identity.digest,
            allow_leaf_rebuild=False,
        )
        pending_collect = _completed_collection_bundle(
            run_dir,
            loop_state,
            design_contract,
        )
        if pending_collect is None:
            pending_collect = _kick_collect(
                start_iteration,
                learner_identity.path,
                learner_identity.digest,
            )

        for it in range(start_iteration, int(args.iterations)):
            t0 = time.time()
            assert pending_collect is not None
            collect_bundle = pending_collect
            shard_path: Path = collect_bundle["shard"]
            writer: CompactShardWriter = collect_bundle["writer"]
            behavior_before = _collection_behavior_identity(collect_bundle)
            incumbent_before = identity
            incumbent_path = ckpt
            learner_before = learner_identity
            rehearsal_record: Optional[dict[str, Any]] = precollect_rehearsals.pop(
                int(it), None
            )

            # SERIALIZE (no AZ-style overlap): collect → train → hard-gate →
            # then start next collect on the *new* digest. Prefetching collect
            # N+1 on old weights caused digest FAIL-CLOSED storms and wasted
            # farm time on a stale build.
            _maybe_apply_live_pool(learner_before.path, learner_before.digest)

            # Collection leaves are large, replicated model processes. No
            # inference work is required during either expert rehearsal or
            # replay-window training, so release the entire farm before *any*
            # corpus packing. The previous ordering stopped leaves only after
            # rehearsal and unnecessarily overlapped ~20 GiB of inference
            # replicas with the large expert CPU pack.
            leaves_suspended_for_train = bool(leaf.remote_channel is not None)
            if leaves_suspended_for_train:
                print(
                    f"[pure_rl] suspend leaf farm before rehearsal/replay "
                    f"iter={it} servers={len(leaf.procs)}",
                    flush=True,
                )
                leaf.stop()
                collected_objects, heap_trimmed = release_process_heap()
                print(
                    f"[pure_rl] leaf memory released before corpus packing "
                    f"iter={it} gc={collected_objects} "
                    f"malloc_trim={int(heap_trimmed)}",
                    flush=True,
                )

            # The ladder pass is durable and happens before loading the large
            # replay window into host RAM.  Recovery checks the receipt/orphan
            # checkpoint first, so a reboot can never repeat a completed pass
            # or silently switch to a newly advanced rolling-data pointer.
            forced_rehearsal = int(args.expert_rehearsal_force_before) == int(it)
            if rehearsal_record is None and (
                forced_rehearsal
                or rehearsal_due(it, int(args.expert_rehearsal_every))
            ):
                if forced_rehearsal:
                    print(
                        f"[pure_rl] forced expert rehearsal boundary iter={it}",
                        flush=True,
                    )
                learner_before, rehearsal_record = _prepare_expert_rehearsal(
                    it, learner_before
                )
                learner_identity = learner_before

            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            if out_ckpt.exists():
                raise RuntimeError(
                    f"refusing to overwrite immutable candidate: {out_ckpt}"
                )

            dataset = _dataset_from_replay_window(
                run_dir,
                it,
                initial_replay_shards=initial_replay_shards,
            )
            n_train_sequences = len(dataset.sequences)
            train_metrics = {
                "mean_advantage": 0.0,
                "raw_advantage_mean": 0.0,
                "raw_advantage_std": 0.0,
                "raw_advantage_mean_abs": 0.0,
                "normalized_advantage_mean": 0.0,
                "normalized_advantage_std": 0.0,
                "awr_weight_mean": 0.0,
                "awr_weight_p50": 0.0,
                "awr_weight_p95": 0.0,
                "awr_weight_clip_frac": 0.0,
                "awr_effective_sample_size": 0.0,
                "awr_effective_sample_fraction": 0.0,
                "policy_selected_nll": 0.0,
                "target_value_mean": 0.0,
                "policy_acc": 0.0,
                "policy_prev_agreement": 0.0,
                "n_sequences": n_train_sequences,
                "collect_temperature": _collect_temperature(args, it),
            }
            if not n_train_sequences:
                del dataset
                release_process_heap()
                raise RuntimeError(
                    f"collect iter={it} passed retention checks but produced no "
                    "trainable sequences"
                )
            print(
                f"[pure_rl] train begin iter={it} seqs={n_train_sequences} "
                f"behavior={behavior_before.digest[:19]}… "
                f"learner_parent={learner_before.digest[:19]}… "
                f"(promotion precedes any weight publish)",
                flush=True,
            )
            decision_cap, batch_schedule_phase = _scheduled_train_decision_cap(
                it,
                steady_cap=int(args.train_max_decisions_per_batch),
                warmup_cap=int(args.train_warmup_max_decisions_per_batch),
                warmup_iterations=int(args.train_warmup_iterations),
            )
            train_cfg = TrainConfig.pure_rl_defaults(
                epochs=max(1, args.train_epochs),
                seed=args.seed + it,
                amp=train_dev.type == "cuda",
                games_per_batch=int(args.train_games_per_batch),
                max_decisions_per_batch=decision_cap,
                aux_loss_weight=float(args.archetype_aux_loss_weight),
                opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
                lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
                prize_race_loss_weight=float(args.prize_race_loss_weight),
                alakazam_guide_loss_weight=float(
                    args.alakazam_guide_loss_weight
                ),
            )
            train_metrics["games_per_batch"] = int(train_cfg.games_per_batch)
            train_metrics["max_decisions_per_batch"] = int(
                train_cfg.max_decisions_per_batch
            )
            train_metrics["batch_schedule_phase"] = batch_schedule_phase
            train_metrics["batch_schedule_warmup_iterations"] = int(
                args.train_warmup_iterations
            )
            print(
                f"[pure_rl] epoch batch contract iter={it} "
                f"games_cap={train_cfg.games_per_batch} "
                f"decisions_cap={train_cfg.max_decisions_per_batch} "
                f"schedule={batch_schedule_phase}",
                flush=True,
            )
            try:
                result = rl_train_step(
                    dataset,
                    base_ckpt=learner_before.path,
                    out_run_name=f"{args.run_name}.iter{it:05d}",
                    archetype_id=(
                        "core"
                        if args.mode == "core"
                        else str(args.specialist_archetype)
                    ),
                    epochs=max(1, args.train_epochs),
                    device=train_dev,
                    cfg=train_cfg,
                    seed=args.seed + it,
                    output_path=out_ckpt,
                    parent_digest=learner_before.digest,
                    training_provenance={
                        "pure_rl": True,
                        "iteration": it,
                        "mode": args.mode,
                        "shard": str(shard_path),
                        "behavior_checkpoint": behavior_before.as_dict(),
                        "learner_parent": learner_before.as_dict(),
                        "expert_rehearsal": rehearsal_record,
                        "opponent_pool": [
                            _verified_checkpoint_identity(entry).as_dict()
                            for entry in opponent_pool
                        ],
                        "design_fingerprint": design_fingerprint,
                        "append_only": True,
                    },
                    replace_existing=False,
                    device_resident=bool(args.train_device_resident),
                    device_resident_min_free_gib=float(
                        args.train_device_resident_min_free_gib
                    ),
                )
            finally:
                # A replay window can retain several GiB of trajectories.  It
                # must not overlap promotion, held-out evaluation, or the next
                # worker pool merely because this trainer is long-lived.
                del dataset
                collected_objects, heap_trimmed = release_process_heap()
                print(
                    f"[pure_rl] replay memory released iter={it} "
                    f"seqs={n_train_sequences} gc={collected_objects} "
                    f"malloc_trim={int(heap_trimmed)}",
                    flush=True,
                )
            if leaves_suspended_for_train:
                _rebuild_leaves_if_needed(
                    learner_before.path,
                    learner_before.digest,
                )
                print(
                    f"[pure_rl] leaf farm restored after train iter={it} "
                    f"servers={len(leaf.procs)}",
                    flush=True,
                )
            candidate_path = Path(
                result.get("candidate_path")
                or result.get("latest_path")
                or out_ckpt
            )
            candidate = CheckpointIdentity.from_path(candidate_path)
            _verify_learner_lineage(
                result, candidate=candidate, parent=learner_before
            )
            m = result.get("metrics") or {}
            if hasattr(m, "__dict__"):
                m = (
                    asdict(m)
                    if hasattr(m, "__dataclass_fields__")
                    else dict(m.__dict__)
                )
            for key in tuple(train_metrics):
                if key in m and key not in ("n_sequences", "collect_temperature"):
                    train_metrics[key] = float(m.get(key) or 0.0)
            train_metrics["optimizer_state_restored"] = bool(
                result.get("optimizer_state_restored", False)
            )
            train_metrics["global_step"] = int(result.get("step", 0))
            train_metrics["epochs_ran"] = int(result.get("epochs_ran", 0))
            train_metrics["awr_baseline_mode"] = str(
                result.get("awr_baseline_mode") or "unknown"
            )

            adv_hist.append(float(train_metrics["raw_advantage_mean_abs"]))
            agr_hist.append(
                float(train_metrics.get("policy_prev_agreement") or 0.0)
            )
            abort = evaluate_aborts(
                mean_advantages=adv_hist,
                advantage_mean_abs=adv_hist,
                policy_prev_agreements=agr_hist,
                k=3,
            )

            promotion_rows: list[dict[str, Any]] = []
            if abort.abort:
                promotion_report = {
                    "passed": False,
                    "valid": False,
                    "skipped": True,
                    "reason": f"abort:{abort.reason}",
                    "candidate": candidate.as_dict(),
                    "incumbent": incumbent_before.as_dict(),
                }
            else:
                print(
                    f"[pure_rl] promotion begin iter={it} "
                    f"candidate={candidate.digest[:19]}… "
                    f"incumbent={incumbent_before.digest[:19]}…",
                    flush=True,
                )
                promotion_report, promotion_rows = _promotion_eval(
                    candidate=candidate,
                    incumbent=incumbent_before,
                    decks=measurement_decks,
                    n_games=args.promotion_games,
                    n_workers=args.promotion_workers,
                    threshold=args.promotion_threshold,
                    confidence=args.promotion_confidence,
                    bootstrap_resamples=args.promotion_bootstrap_resamples,
                    seed=args.seed + 7_000_000 + it * 10_000,
                    game_timeout_s=args.game_timeout_s,
                    model_generation=it + 1,
                )
                promotion_report["skipped"] = False

            promoted = bool(promotion_report.get("passed")) and not abort.abort
            candidate_safety_ok, candidate_safety_reason = carry_learner_candidate(
                promotion_report,
                abort=bool(abort.abort),
                minimum_head_to_head_wr=float(args.continuous_learner_min_wr),
            )
            prior_heldout_champion_evidence = dict(heldout_champion_evidence)
            weight_gate_proof: Optional[dict[str, Any]] = None
            heldout_publish_proof: Optional[dict[str, Any]] = None
            heldout_rollback_proof: Optional[dict[str, Any]] = None
            publish_version_base = (it + 1) * 10
            if promoted:
                ckpt = candidate_path
                identity = candidate
                opponent_pool.append(candidate)
                pool_n = max(
                    1, int(getattr(config.PURE_RL, "opponent_pool_size", 6))
                )
                del opponent_pool[:-pool_n]
                print(
                    f"[pure_rl] BETWEEN_ITER_HARD_GATE begin iter={it} "
                    f"digest={identity.digest[:19]}… version={publish_version_base + 1} "
                    f"(before next collect)",
                    flush=True,
                )
                weight_gate_proof = _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=remote_farm,
                    ckpt=ckpt,
                    digest=identity.digest,
                    version=publish_version_base + 1,
                    required_endpoints=endpoints,
                )
                promotion_report["deployment"] = weight_gate_proof
                print(
                    f"[pure_rl] PROMOTED iter={it} "
                    f"wr={promotion_report.get('wr')} "
                    f"lower={promotion_report.get('interval_lower')}",
                    flush=True,
                )
            else:
                ckpt = incumbent_path
                identity = incumbent_before
                print(
                    f"[pure_rl] REJECTED iter={it} candidate retained={candidate_path} "
                    f"incumbent={identity.digest[:19]}… "
                    f"reason={promotion_report.get('reason') or promotion_report.get('failures')}",
                    flush=True,
                )

            # Always evaluate the candidate, never the incumbent substituted
            # after rejection.  A rejected candidate is published only for this
            # drained greedy wave and the protected champion is restored in a
            # finally block before any next-iteration collection can begin.
            if not promoted:
                heldout_publish_proof = _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=(remote_farm if args.heldout_remotes else None),
                    ckpt=candidate.path,
                    digest=candidate.digest,
                    version=publish_version_base + 2,
                    required_endpoints=(endpoints if args.heldout_remotes else []),
                )
            try:
                heldout_rows, heldout_audit = _heldout_eval(
                    ckpt=Path(candidate.path),
                    digest=candidate.digest,
                    n_games=args.heldout_games,
                    decks=measurement_decks,
                    official_specs=heldout_specs,
                    seed=args.seed + 19_000_000 + it * 100_000,
                    game_timeout_s=args.game_timeout_s,
                    n_workers=hw.sim_workers,
                    leaf_channel=leaf.remote_channel,
                    remote_farm=(remote_farm if args.heldout_remotes else None),
                    worker_play=worker_play,
                    worker_self_play=worker_self_play,
                    mode=args.mode,
                    allow_remote_play=bool(args.heldout_remotes),
                    iteration=int(it),
                    gate_wr=args.gate_wr,
                    opponent_ids=active_gate_ids,
                    stage_label=(
                        "heldout:strong_public_gate"
                        if active_gate is not None
                        else "heldout"
                    ),
                )
            finally:
                if not promoted:
                    heldout_rollback_proof = _hard_gate_publish_weights(
                        leaf=leaf,
                        remote_farm=(remote_farm if args.heldout_remotes else None),
                        ckpt=incumbent_before.path,
                        digest=incumbent_before.digest,
                        version=publish_version_base + 3,
                        required_endpoints=(
                            endpoints if args.heldout_remotes else []
                        ),
                    )
            gate = aggregate_heldout_wr(
                heldout_rows,
                target_wr=args.gate_wr,
                min_games=args.heldout_games,
                official_ids=active_gate_ids,
                min_games_per_opponent=(
                    int(args.heldout_games) // len(active_gate_ids)
                ),
                per_opponent_floor=args.heldout_per_opponent_floor,
            )
            active_gate_result: Optional[dict[str, Any]] = None
            if active_gate_contract is not None:
                active_gate_result = build_active_gate_result(
                    contract=active_gate_contract,
                    checkpoint=candidate.path,
                    checkpoint_digest=candidate.digest,
                    iteration=int(it),
                    gate_rows=heldout_rows,
                    gate_audit=heldout_audit,
                    gate_seed=args.seed + 19_000_000 + it * 100_000,
                )
                gate.win_rate = float(active_gate_result["skill_weighted_wr"])
                gate.confidence_lower = float(
                    active_gate_result["confidence_lower"]
                )
                gate.confidence_upper = float(
                    active_gate_result["confidence_upper"]
                )
                gate.passed = bool(active_gate_result["passed"])
                gate.reason = "ok" if gate.passed else "active_gate_criteria_failed"
                raw_heldout_gate = dict(active_gate_result)
            else:
                raw_heldout_gate = asdict(gate)
            heldout_champion_updated = False
            learner_exploration = {
                "eligible": False,
                "reason": "heldout_contract_audit_failed",
            }
            if bool(heldout_audit.get("passed")):
                candidate_heldout_evidence = {
                    "evidence_schema": 2,
                    "gate_id": (
                        str(active_gate["id"]) if active_gate is not None else None
                    ),
                    "iteration": int(it),
                    "checkpoint_digest": candidate.digest,
                    "games": int(gate.games),
                    "win_rate": float(gate.win_rate),
                    "confidence_lower": float(gate.confidence_lower),
                    "confidence_upper": float(gate.confidence_upper),
                    "per_opponent": json.loads(json.dumps(gate.per_opponent)),
                    "audit": dict(heldout_audit),
                }
                prior_active_evidence = (
                    heldout_champion_evidence
                    if active_gate is None
                    or str(heldout_champion_evidence.get("gate_id") or "")
                    == str(active_gate["id"])
                    else {}
                )
                candidate_rank = heldout_goal_rank(
                    candidate_heldout_evidence,
                    official_ids=active_gate_ids,
                    per_opponent_floor=float(args.heldout_per_opponent_floor),
                )
                prior_rank = heldout_goal_rank(
                    prior_active_evidence,
                    official_ids=active_gate_ids,
                    per_opponent_floor=float(args.heldout_per_opponent_floor),
                )
                if candidate_safety_ok and (
                    not prior_active_evidence or candidate_rank > prior_rank
                ):
                    heldout_champion_identity = candidate
                    heldout_champion_evidence = candidate_heldout_evidence
                    heldout_champion_updated = True
                    print(
                        f"[pure_rl] HELDOUT_CHAMPION iter={it} "
                        f"digest={candidate.digest[:19]}… "
                        f"wr={gate.win_rate:.3f} lower={gate.confidence_lower:.3f}",
                        flush=True,
                    )
                elif candidate_safety_ok and prior_active_evidence:
                    learner_exploration = heldout_exploration_decision(
                        candidate_heldout_evidence,
                        prior_active_evidence,
                        official_ids=active_gate_ids,
                        per_opponent_floor=float(
                            args.heldout_per_opponent_floor
                        ),
                    )
                    if bool(learner_exploration.get("eligible")):
                        print(
                            f"[pure_rl] CONTINUOUS_LEARNER iter={it} "
                            f"digest={candidate.digest[:19]}… "
                            "heldout_anchor_unchanged=1 "
                            f"progress_gain={float(learner_exploration['clipped_progress_gain']):.4f}",
                            flush=True,
                        )
            # Keep the exact heldout record protected, but let the separate
            # learner accumulate every H2H-safe step whose formal heldout
            # execution passed the checkpoint/seat/opponent audit.  Requiring
            # a new per-opponent record here reset iterations 1-4 to iter 0 and
            # prevented cumulative RL toward the strong-public gate.
            carry_candidate, learner_carry_reason = (
                _continuous_learner_carry_decision(
                    candidate_safety_ok=bool(candidate_safety_ok),
                    candidate_safety_reason=str(candidate_safety_reason),
                    heldout_audit_ok=bool(heldout_audit.get("passed")),
                    promoted=bool(promoted),
                )
            )
            if carry_candidate:
                learner_after = candidate
            else:
                # Reject only this step.  The last safety-approved learner is
                # the rollback target; the protected heldout champion remains
                # independently available and must not erase learner history.
                learner_after = learner_before
            learner_identity = learner_after
            promotion_report["continuous_learner"] = {
                "carried_candidate": bool(carry_candidate),
                "candidate_head_to_head_safe": bool(candidate_safety_ok),
                "reason": learner_carry_reason,
                "selection_objective": (
                    "protected_heldout_best_plus_h2h_safe_continuous_learner_v3"
                ),
                "exploration": learner_exploration,
                "minimum_head_to_head_wr": float(args.continuous_learner_min_wr),
                "learner_before": learner_before.as_dict(),
                "learner_after": learner_after.as_dict(),
            }
            if not bool(heldout_audit.get("passed")):
                gate.passed = False
                gate.reason = "heldout_contract_audit_failed"
            elif not promoted:
                gate.passed = False
                gate.reason = "candidate_not_promoted"
            if active_gate_result is not None:
                active_gate_result["pipeline_gate_passed"] = bool(gate.passed)
                active_gate_result["pipeline_gate_reason"] = str(gate.reason)
                active_gate_result["promotion_passed"] = bool(promoted)
                active_gate_result["candidate_safety_passed"] = bool(
                    candidate_safety_ok
                )
            # The protected champion remains the promotion/rollback identity,
            # while the safety-approved continuous learner is the next behavior
            # policy. Publish it only after promotion and heldout work is drained
            # so each immutable collection shard uses one exact digest.
            collection_publish_proof: Optional[dict[str, Any]] = None
            if not gate.passed and it + 1 < int(args.iterations):
                collection_publish_proof = _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=remote_farm,
                    ckpt=learner_after.path,
                    digest=learner_after.digest,
                    version=publish_version_base + 4,
                    required_endpoints=endpoints,
                )
                collection_publish_proof["role"] = "continuous_learner"
            eval_payload = {
                "iteration": it,
                "candidate": candidate.as_dict(),
                "incumbent_before": incumbent_before.as_dict(),
                "incumbent_after": identity.as_dict(),
                "learner_before": learner_before.as_dict(),
                "learner_after": learner_after.as_dict(),
                "expert_rehearsal": rehearsal_record,
                "promotion": promotion_report,
                "promotion_rows": promotion_rows,
                "raw_heldout_gate": raw_heldout_gate,
                "active_gate_result": active_gate_result,
                "heldout_gate": asdict(gate),
                "heldout_audit": heldout_audit,
                "heldout_candidate": candidate.as_dict(),
                "heldout_champion": heldout_champion_identity.as_dict(),
                "heldout_champion_evidence": heldout_champion_evidence,
                "heldout_champion_updated": heldout_champion_updated,
                "heldout_publish": heldout_publish_proof,
                "heldout_rollback": heldout_rollback_proof,
                "next_collection_publish": collection_publish_proof,
                "heldout_rows": heldout_rows,
            }
            _write_json_exclusive(
                run_dir / "eval" / f"iter_{it:05d}.json", eval_payload
            )
            collect_stats = dict(collect_bundle["stats"])
            collect_elapsed = max(
                float(collect_stats.get("collect_elapsed_sec") or 0.0), 1e-6
            )
            completed_at = time.time()
            elapsed = max(completed_at - t0, 1e-6)
            iteration_started_at = float(
                collect_bundle.get("started_at") or (completed_at - collect_elapsed - elapsed)
            )
            iteration_wall_sec = max(completed_at - iteration_started_at, 1e-6)
            metrics = IterationMetrics(
                iteration=it,
                stage=stage.stage.value,
                games=writer.n_games,
                decisions=writer.n_decisions,
                games_per_sec=writer.n_games / collect_elapsed,
                decisions_per_sec=writer.n_decisions / collect_elapsed,
                games_per_hour=(writer.n_games / collect_elapsed) * 3600.0,
                mean_return=float(train_metrics["target_value_mean"]),
                mean_advantage=float(train_metrics["mean_advantage"]),
                raw_advantage_mean=float(train_metrics["raw_advantage_mean"]),
                raw_advantage_std=float(train_metrics["raw_advantage_std"]),
                raw_advantage_mean_abs=float(
                    train_metrics["raw_advantage_mean_abs"]
                ),
                normalized_advantage_mean=float(
                    train_metrics["normalized_advantage_mean"]
                ),
                normalized_advantage_std=float(
                    train_metrics["normalized_advantage_std"]
                ),
                awr_weight_mean=float(train_metrics["awr_weight_mean"]),
                awr_weight_p50=float(train_metrics["awr_weight_p50"]),
                awr_weight_p95=float(train_metrics["awr_weight_p95"]),
                awr_weight_clip_frac=float(train_metrics["awr_weight_clip_frac"]),
                awr_effective_sample_size=float(
                    train_metrics["awr_effective_sample_size"]
                ),
                awr_effective_sample_fraction=float(
                    train_metrics["awr_effective_sample_fraction"]
                ),
                policy_selected_nll=float(train_metrics["policy_selected_nll"]),
                policy_accuracy=float(train_metrics["policy_acc"]),
                policy_prev_agreement=float(agr_hist[-1]),
                self_distill_flag=abort.self_distill_flag,
                heldout_wr=gate.win_rate,
                heldout_games=gate.games,
                gate_passed=gate.passed,
                extra={
                    "abort": asdict(abort),
                    # elapsed_sec is retained for compatibility with existing
                    # reports. It historically measured only post-collection
                    # work, so the explicit wall metric is authoritative.
                    "elapsed_sec": elapsed,
                    "post_collect_elapsed_sec": elapsed,
                    "iteration_started_at": iteration_started_at,
                    "iteration_completed_at": completed_at,
                    "iteration_wall_sec": iteration_wall_sec,
                    "collect_stats": collect_stats,
                    "checkpoint": str(ckpt),
                    "candidate_checkpoint": candidate.as_dict(),
                    "incumbent_before": incumbent_before.as_dict(),
                    "learner_before": learner_before.as_dict(),
                    "learner_after": learner_after.as_dict(),
                    "expert_rehearsal": rehearsal_record,
                    "promotion": promotion_report,
                    "raw_heldout_gate": raw_heldout_gate,
                    "active_gate_result": active_gate_result,
                    "heldout_gate": asdict(gate),
                    "heldout_audit": heldout_audit,
                    "heldout_champion": heldout_champion_identity.as_dict(),
                    "heldout_champion_evidence": heldout_champion_evidence,
                    "heldout_champion_updated": heldout_champion_updated,
                    "heldout_publish": heldout_publish_proof,
                    "heldout_rollback": heldout_rollback_proof,
                    "next_collection_publish": collection_publish_proof,
                    "train_metrics": train_metrics,
                    "remote_workers": (
                        remote_farm.total_workers if remote_farm else 0
                    ),
                    "remote_endpoints": endpoints,
                    "n_train_sequences": train_metrics["n_sequences"],
                    "between_iter_hard_gate": weight_gate_proof
                    or {"skipped": not promoted},
                },
            )
            _write_metrics(run_dir, it, metrics)

            # Commit all immutable artifacts, then advance the mutable pointer.
            next_state = json.loads(json.dumps(loop_state))
            next_state.update(
                {
                    "next_iteration": it + 1,
                    "last_completed_iteration": it,
                    "champion": identity.as_dict(),
                    "heldout_champion": heldout_champion_identity.as_dict(),
                    "heldout_champion_evidence": heldout_champion_evidence,
                    "learner": learner_after.as_dict(),
                    "opponent_pool": [
                        _verified_checkpoint_identity(entry).as_dict()
                        for entry in opponent_pool
                    ],
                    "raw_advantage_history": adv_hist,
                    "policy_prev_agreement_history": agr_hist,
                    "live_pool_last_seq": live_pool_last_seq,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            next_state.setdefault("history", []).append(
                {
                    "iteration": it,
                    "completed": True,
                    "candidate": candidate.as_dict(),
                    "promoted": promoted,
                    "learner_before": learner_before.as_dict(),
                    "learner_after": learner_after.as_dict(),
                    "expert_rehearsal": rehearsal_record,
                    "promotion": promotion_report,
                    "incumbent_after": identity.as_dict(),
                    "raw_heldout_gate": raw_heldout_gate,
                    "active_gate_result": active_gate_result,
                    "heldout_audit": heldout_audit,
                    "heldout_champion": heldout_champion_identity.as_dict(),
                    "heldout_champion_updated": heldout_champion_updated,
                    "next_collection_publish": collection_publish_proof,
                    "stage_gate": asdict(gate),
                    "metrics": str(
                        run_dir / "metrics" / f"iter_{it:05d}.json"
                    ),
                    "eval": str(run_dir / "eval" / f"iter_{it:05d}.json"),
                }
            )
            _write_json_exclusive(
                run_dir / "commits" / f"iter_{it:05d}.json", next_state
            )
            _atomic_json(run_dir / "loop_state.json", next_state)
            if active_gate_result is not None and active_gate is not None:
                result_pointer = Path(
                    str(active_gate.get("exact_result_pointer") or "")
                ).expanduser()
                if not str(result_pointer):
                    raise RuntimeError("active gate has no exact_result_pointer")
                committed_gate_result = dict(active_gate_result)
                committed_gate_result.update(
                    {
                        "committed": True,
                        "commit": str(
                            run_dir / "commits" / f"iter_{it:05d}.json"
                        ),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                _atomic_json(result_pointer, committed_gate_result)
            _atomic_json(
                run_dir / "iteration_runtime.json",
                {
                    "schema": "poke_bot.iteration_runtime/v1",
                    "iteration": int(it),
                    "phase": "completed",
                    "started_at": iteration_started_at,
                    "completed_at": completed_at,
                    "elapsed_sec": iteration_wall_sec,
                    "updated_at": time.time(),
                },
            )
            loop_state = next_state
            _retain_completed_artifacts(loop_state, it)
            print(
                f"[pure_rl] iter={it} games={metrics.games} "
                f"seqs={train_metrics['n_sequences']} "
                f"awr_w={metrics.awr_weight_mean:.3f} "
                f"heldout_wr={gate.win_rate:.3f} ({gate.games}g) "
                f"promoted={promoted} learner_carry={carry_candidate} "
                f"stage_gate={gate.passed} audit={heldout_audit.get('passed')} "
                f"abort={abort.abort} "
                f"gps={metrics.games_per_sec:.2f}",
                flush=True,
            )
            try:
                from poke_bot.pure_rl.wr_trend import wr_trend_line

                print(
                    f"[pure_rl] {wr_trend_line(run_dir, gate_wr=args.gate_wr)}",
                    flush=True,
                )
            except Exception as exc:  # never let a display helper break training
                print(f"[pure_rl] wr_trend WARN: {exc!r}", flush=True)
            if gate.passed:
                marker = _ensure_terminal_gate_marker(run_dir, loop_state)
                if marker is None:
                    raise RuntimeError(
                        "committed gate passed but terminal marker payload was absent"
                    )
                print(f"[pure_rl] {marker.name}", flush=True)
                break
            next_it = it + 1
            if next_it < int(args.iterations):
                print(
                    f"[pure_rl] kick collect iter={next_it} on continuous learner "
                    f"digest={learner_after.digest[:19]}…",
                    flush=True,
                )
                pending_collect = _kick_collect(
                    next_it,
                    learner_after.path,
                    learner_after.digest,
                )
            else:
                pending_collect = None
        return 0
    finally:
        expert_cache.release()
        leaf.stop()
        if remote_farm is not None:
            try:
                remote_farm.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.allow_single_gpu:
        os.environ["PURE_RL_ALLOW_SINGLE_GPU"] = "1"
    if args.smoke:
        return run_smoke_loop(args)
    return run_full_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
